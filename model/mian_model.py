import torch
import torch.nn as nn
from transformers import BertModel, RobertaModel
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import kl_divergence


class CVAE(nn.Module):
    def __init__(self, hps):
        super(CVAE, self).__init__()
        self.hps = hps
        self.z_mean = nn.Linear(2*hps.hidden_dim, hps.gmm_size)
        self.z_log_var = nn.Linear(2*hps.hidden_dim, hps.gmm_size)

        self.condition_z_mean = nn.Linear(3*hps.hidden_dim, hps.gmm_size)
        self.condition_z_log_var = nn.Linear(3*hps.hidden_dim, hps.gmm_size)
        self.relu = nn.ReLU()

    def forward(self, events, contexts):
        # print(events.size())
        event_pairs = [torch.cat((events[:, i:i+1, :], events[:, i+1:i+2, :]), -1) for i in range(len(events[0])-1)]
        event_pairs = torch.cat(tuple(event_pairs), 1)
        z_mean = self.z_mean(event_pairs)
        z_log_var = self.z_log_var(event_pairs)
        if not self.hps.pretrain:
            condition_events = torch.cat((event_pairs, contexts), -1)
            q_z_mean = self.condition_z_mean(condition_events)
            q_z_log_var = self.condition_z_log_var(condition_events)

            tmp_kl_loss = torch.FloatTensor([0]).to(events.device)
            for i in range(z_mean.size()[0]):
                for j in range(z_log_var.size()[1]):
                    p = MultivariateNormal(z_mean[i, j], torch.diag(z_log_var[i, j].exp()))
                    q = MultivariateNormal(q_z_mean[i, j], torch.diag(q_z_log_var[i, j].exp()))
                    tmp_kl_loss += kl_divergence(p, q)
            kl_loss = tmp_kl_loss / (z_mean.size()[0] * z_log_var.size()[1])   

        # kl_loss = 0.5 * (q_z_log_var - z_log_var + (z_log_var.exp() + (z_mean - q_z_mean).pow(2)) / q_z_log_var.exp() - 1)
        # kl_loss = torch.mean(torch.mean(torch.sum(kl_loss, -1), -1), 0)

            return z_mean, z_log_var, q_z_mean, q_z_log_var, kl_loss
        else:
            return z_mean, z_log_var, z_mean, z_log_var, torch.FloatTensor([0]).to(events.device)


class Event_Encoder(nn.Module):
    def __init__(self, hps):
        super(Event_Encoder, self).__init__()
        self.hps = hps
        if 'roberta' in hps.transformer_dir:
            self.encoder = RobertaModel.from_pretrained(hps.transformer_dir)
        else:
            self.encoder = BertModel.from_pretrained(hps.transformer_dir)
        self.linear = nn.Linear(self.encoder.config.hidden_size, hps.hidden_dim)

    def forward(self, event_ids, attention_mask):
        output = self.encoder(input_ids=event_ids, attention_mask=attention_mask)
        tokens = output[0]
        if 'roberta' in self.hps.transformer_dir:
            sep_indexes = (event_ids == 2).nonzero()
        elif 'bert' in self.hps.transformer_dir:
            sep_indexes = (event_ids == 102).nonzero()
        else:
            sep_indexes = (event_ids == 102).nonzero()
        
        seps = self.linear(tokens[sep_indexes[:, 0], sep_indexes[:, 1], :])
        e1, e2, e3, e4, e5 = seps[::5, :], seps[1::5, :], seps[2::5, :], seps[3::5, :], seps[4::5, :]
        e1 = e1.unsqueeze(1)
        e2 = e2.unsqueeze(1)
        e3 = e3.unsqueeze(1)
        e4 = e4.unsqueeze(1)
        e5 = e5.unsqueeze(1)
        events = torch.cat((e1, e2, e3, e4, e5), 1)
        return events, self.encoder


class Context_Encoder(nn.Module):
    def __init__(self, hps):
        super(Context_Encoder, self).__init__()
        self.hps = hps
        # self.encoder = RobertaModel.from_pretrained(hps.transformer_dir)
        if 'large' in hps.transformer_dir:
            self.linear = nn.Linear(1024, hps.hidden_dim)
        else:
            self.linear = nn.Linear(768, hps.hidden_dim)

    def forward(self, context_ids, attention_mask, encoder):
        output = encoder(input_ids=context_ids, attention_mask=attention_mask)
        tokens = output[0]

        if 'roberta' in self.hps.transformer_dir:
            sep_indexes = (context_ids == 2).nonzero()
        elif 'bert' in self.hps.transformer_dir:
            sep_indexes = (context_ids == 102).nonzero()
        else:
            sep_indexes = (context_ids == 102).nonzero()

        seps = self.linear(tokens[sep_indexes[:, 0], sep_indexes[:, 1], :])
        c1, c2, c3, c4 = seps[::4, :], seps[1::4, :], seps[2::4, :], seps[3::4, :]
        c1 = c1.unsqueeze(1)
        c2 = c2.unsqueeze(1)
        c3 = c3.unsqueeze(1)
        c4 = c4.unsqueeze(1)
        contexts = torch.cat((c1, c2, c3, c4), 1)
        return contexts


class Chain(nn.Module):
    def __init__(self, hps):
        super(Chain, self).__init__()
        self.hps = hps
        self.for_threshold = nn.Bilinear(2*hps.hidden_dim, hps.gmm_size, hps.hidden_dim)
        self.threshold_gate = nn.Bilinear(hps.hidden_dim, hps.hidden_dim, 1)
        self.scene_gate = nn.Bilinear(hps.gmm_size, hps.gmm_size, 1)
        self.joint_distribution = nn.Linear(2*hps.hidden_dim+hps.gmm_size, hps.hidden_dim)
        self.final_prob = nn.Bilinear(hps.hidden_dim, hps.hidden_dim, hps.hidden_dim)
        self.prediction = nn.Linear(hps.hidden_dim, 2)
        self.softmax = nn.Softmax(-1)
        self.gelu = nn.GELU()

    def forward(self, A, B, C, U_AB, U_BC):
        ab = self.for_threshold(torch.cat((A, B), -1), U_AB)
        bc = self.for_threshold(torch.cat((B, C), -1), U_BC)
        ab, bc = self.gelu(ab), self.gelu(bc)
        alpha = self.threshold_gate(ab, bc)
        beta = self.scene_gate(U_AB, U_BC)
        U_BC = U_BC + (alpha + beta) / 2 * U_AB
        P_abu = self.joint_distribution(torch.cat((A, B, U_AB), -1))
        P_bcu = self.joint_distribution(torch.cat((B, C, U_BC), -1))
        P_abu, P_bcu = self.gelu(P_abu), self.gelu(P_bcu)
        P = self.final_prob(P_abu, P_bcu)
        logits = self.prediction(P)
        # probs = self.softmax(logits)
        return logits, alpha, beta


class SRNN(nn.Module):
    def __init__(self, hps):
        super(SRNN, self).__init__()
        self.hps = hps
        self.scene_valve_0 = nn.Linear(hps.gmm_size, hps.hidden_dim, bias=True)
        self.scene_valve_1 = nn.Linear(hps.gmm_size, hps.hidden_dim, bias=True)

        self.fusion_gate = nn.Linear(hps.hidden_dim*2, hps.hidden_dim, bias=True)

        self.threshold_valve_0 = nn.Linear(hps.gmm_size+hps.hidden_dim, hps.hidden_dim, bias=True)
        self.threshold_valve_1 = nn.Linear(hps.gmm_size+hps.hidden_dim, hps.hidden_dim, bias=True)

        self.exofenous_fusion = nn.Linear(hps.hidden_dim, hps.gmm_size, bias=True)

        self.output_gate = nn.Linear(hps.gmm_size*2, hps.gmm_size, bias=True)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    
    def forward(self, A, B, C, UAB, UBC):
        alpha_t = self.sigmoid(self.scene_valve_0(UAB)-self.scene_valve_1(UBC))
        A_t = self.tanh(A + B + self.fusion_gate(torch.cat((A, B), -1)))
        B_t = self.tanh(B + C + self.fusion_gate(torch.cat((B, C), -1)))

        threshold1 = self.threshold_valve_0(torch.cat((UBC, B_t), -1))
        threshold2 = self.threshold_valve_1(torch.cat((UAB, A_t), -1))
        beta_t = self.sigmoid((threshold2 - threshold1)*(1.0-alpha_t))

        tmp_e = UBC + (alpha_t + beta_t) / 2 * UAB
        E_t = self.tanh(self.exofenous_fusion(tmp_e))
        UAB_t = self.tanh(self.output_gate(torch.cat((UAB, E_t), -1)))
        # UAB_t = E_t

        return alpha_t, beta_t, A_t, B_t, UAB, UAB_t, E_t


class ReCo(nn.Module):
    def __init__(self, hps):
        super(ReCo, self).__init__()
        self.hps = hps
        self.cvae = CVAE(hps)
        self.srnn = SRNN(hps)
        self.event_encoder = Event_Encoder(hps)
        self.context_encoder = Context_Encoder(hps)
        self.ps = nn.Linear(hps.hidden_dim, 2, bias=True)
        self.pt = nn.Linear(hps.hidden_dim, 2, bias=True)
        self.p1 = nn.Bilinear(hps.hidden_dim, hps.hidden_dim, hps.hidden_dim, bias=True)
        self.p2 = nn.Bilinear(hps.hidden_dim, hps.hidden_dim, hps.hidden_dim, bias=True)
        self.pc = nn.Bilinear(hps.hidden_dim, hps.hidden_dim, 2, bias=True)
        self.softmax = nn.Softmax(-1)
        self.tanh = nn.Tanh()
        self.loss_func = nn.CrossEntropyLoss(reduction='mean', weight=torch.FloatTensor([0.6, 0.4]))
        # self.loss_func = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, batch, epslion):
        print(batch[0].size())
        event_ids, event_id_mask, context_id, context_id_mask, labels, threshold_labels, scene_labels = batch

        events, encoder = self.event_encoder(event_ids, event_id_mask)
        if not self.hps.pretrain:
            contexts = self.context_encoder(context_id, context_id_mask, encoder)
        else:
            contexts = events

        if self.hps.mode == 'train':
            _, _, z_mean, z_log_var, kl_loss = self.cvae(events, contexts)
        else:
            z_mean, z_log_var, _, _, kl_loss = self.cvae(events, contexts)

        z = z_mean + epslion * z_log_var.exp()

        # Structural Causal Recurrent Unit
        alpha, beta, uab, a, b, e = [], [], [], [], [], []
        new_UAB = z[:, 0]
        new_UBC = z[:, 1]
        A_t = events[:, 0]
        B_t = events[:, 0]
        
        for i in range(2, 5):
            
            C = events[:, i]
            UAB = new_UAB
            UBC = new_UBC
            alpha_t, beta_t, A_t, B_t, UAB, UAB_t, E_t = self.srnn(A_t, B_t, C, UAB, UBC)
            if i < 4:
                new_UAB, new_UBC = UAB_t, z[:, i]
            alpha.append(alpha_t.unsqueeze(1))
            beta.append(beta_t.unsqueeze(1))
            uab.append(UAB.unsqueeze(1))
            a.append(A_t.unsqueeze(1))
            b.append(B_t.unsqueeze(1))
            e.append(E_t.unsqueeze(1))

        # Prediction Method
        alpha = torch.cat(tuple(alpha), 1)
        beta = torch.cat(tuple(beta), 1)
        uab = torch.cat(tuple(uab), 1)
        a = torch.cat(tuple(a), 1)
        b = torch.cat(tuple(b), 1)
        e = torch.cat(tuple(e), 1)

        ps = self.softmax(self.ps(alpha))
        pt = self.softmax(self.pt(beta))

        p1 = self.tanh(self.p1(a, uab))
        p2 = self.tanh(self.p2(b, e))
        pc = self.pc(p1, p2)

        loss = compute_training_loss(self.hps, ps, pt, pc, labels, scene_labels, threshold_labels, self.loss_func, kl_loss)

        # return loss, pc, alpha, beta
        return loss, pc, ps, pt


def compute_training_loss(hps, ps, pt, pc, labels, scene_labels, threshold_labels, loss_func, kl_loss):
    softmax = nn.Sigmoid()
    PC = []
    PS = []
    PT = []
    chain_labels = []
    PCS = []
    for i in range(ps.size()[0]):
        label = labels[i].cpu().item()
        if label == 5:
            PC.append(pc[i])
            PS.append(ps[i, :, 0])
            PT.append(pt[i, :, 0])
            PCS.append(pc[i, :, 1])
            chain_labels += [1, 1, 1]
        elif label == 2:
            PC.append(pc[i, 0:1])
            PS.append(ps[i, 0, 1:2] if scene_labels[i].item() == 1 else ps[i, 0, 0:1])
            PT.append(pt[i, 0, 1:2] if threshold_labels[i].item() == 1 else pt[i, 0, 0:1])
            PCS.append(pc[i, :1, 0])
            chain_labels += [0]

        elif label == 3:
            PC.append(pc[i, 0:2])
            PS.append(ps[i, 0, 0:1])
            PS.append(ps[i, 1, 1:2] if scene_labels[i].item() == 1 else ps[i, 1, 0:1])
            PT.append(pt[i, 0, 0:1])
            PT.append(pt[i, 0, 0:1] if threshold_labels[i].item() == 1 else pt[i, 1, 0:1])
            PCS.append(pc[i, :1, 1])
            PCS.append(pc[i, 1:2, 0])
            chain_labels += [1, 0]

        else:
            PC.append(pc[i, 0:3])
            PS.append(ps[i, :2, 0])
            PS.append(ps[i, 2, 1:2] if scene_labels[i].item() == 1 else ps[i, 2, 0:1])
            PT.append(pt[i, :2, 0])
            PT.append(pt[i, 2, 1:2] if threshold_labels[i].item() == 1 else pt[i, 2, 0:1])
            PCS.append(pc[i, :2, 1])
            PCS.append(pc[i, 2:3, 0])
            chain_labels += [1, 1, 0]
    PC = torch.cat(tuple(PC), 0)
    PS = torch.cat(tuple(PS), 0)
    PT = torch.cat(tuple(PT), 0)
    PCS = torch.cat(tuple(PCS), 0)
    chain_labels = torch.LongTensor(chain_labels).to(PC.device) if hps.cuda else torch.LongTensor(chain_labels)

    cross_entropy = loss_func(PC, chain_labels)
    if not hps.pretrain:
        logic_loss = torch.mean(torch.abs(torch.log(PS * PT) - torch.log(softmax(PCS))), 0)
        loss = cross_entropy.squeeze() + hps.lambda2 * kl_loss.squeeze() + hps.lambda1 * logic_loss.squeeze()
        return loss
    else:
        return cross_entropy

