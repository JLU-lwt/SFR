import time
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from beam_search import BeamHypotheses
from bfgs import  bfgs_F, bfgs_FG
from data_factory_test import de_tokenize
from embedding import NodeState_Embedding,Y_Embedding,NeighbourState_Embedding
from SetTransformer import SetTransformer
from NNS_metrics import cal_metrics
import pandas as pd
import sympy as sp
from sympy.core.rules import Transform
from sympy import sympify, Float, Symbol
import transformers 

def constants_to_placeholder(s,symbol="CONSTANT"):
    sympy_expr = sympify(s)  # self.infix_to_sympy("(" + s + ")")
    sympy_expr = sympy_expr.xreplace(
        Transform(
            lambda x: Symbol(symbol, real=True, nonzero=True),
            lambda x: isinstance(x, Float),
        )
    )
    return sympy_expr

class Model(pl.LightningModule):
    def __init__(self,cfg,metadata):
        super(Model, self).__init__()
        self.cfg = cfg
        self.metadata=metadata

        self.node_state_embedding = NodeState_Embedding(cfg)
        self.neighbour_state_embedding = NeighbourState_Embedding(cfg)
        self.y_embedding = Y_Embedding(cfg)

    
        self.token_embedding = nn.Embedding(cfg.output_dim, cfg.dim_hidden)
        self.pos_embedding = nn.Embedding(cfg.length_eq, cfg.dim_hidden)
        self.dropout = nn.Dropout(cfg.dropout)
        self.trg_pad_idx = cfg.trg_pad_idx

        self.encode_set_transformer = SetTransformer(dim_input=self.cfg.dim_hidden,
                                                     num_outputs=self.cfg.num_outputs,
                                                     dim_output=self.cfg.dim_hidden,
                                                     dim_hidden=self.cfg.dim_hidden,
                                                     num_inds=self.cfg.num_inds,
                                                     num_heads=self.cfg.num_heads,
                                                     num_ISABs=self.cfg.num_ISABs,
                                                     num_SABs=self.cfg.num_SABs,
                                                     ln=self.cfg.ln)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.cfg.dim_hidden,
            nhead=self.cfg.num_heads,
            dim_feedforward=self.cfg.dec_pf_dim,
            dropout=self.cfg.dropout,
        )

        self.F_decoder_transformer = nn.TransformerDecoder(decoder_layer, num_layers=self.cfg.num_dec_layers)
        self.G_decoder_transformer = nn.TransformerDecoder(decoder_layer, num_layers=self.cfg.num_dec_layers)

        self.fc_out = nn.Linear(self.cfg.dim_hidden, cfg.output_dim)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).float()

        trg_pad_mask = (
            trg_pad_mask.masked_fill(trg_pad_mask == 0, float("-inf"))
            .masked_fill(trg_pad_mask == 1, float(0.0))
            .type_as(trg)
        )
        trg_len = trg.shape[1]
        mask = (torch.triu(torch.ones(trg_len, trg_len)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
            .type_as(trg)
        )
        return trg_pad_mask, mask


    def forward(self,batch):
        node_state, neighbour_state, info, y, F_token, G_token, _, _, dimension = batch
        node_state_emb = self.node_state_embedding(node_state)
        neighbour_state_emb = self.neighbour_state_embedding(node_state_emb,neighbour_state, info)
        y_emb = self.y_embedding(y)

        enc_emb = node_state_emb + neighbour_state_emb + y_emb
        enc_output = self.encode_set_transformer(enc_emb)
        F_token = F_token.long()
        G_token = G_token.long()
        F_pos_emb = self.pos_embedding(
            torch.arange(0, F_token.shape[1] - 1)
            .unsqueeze(0)
            .repeat(F_token.shape[0], 1)
            .type_as(F_token)
        )
        F_token_emb_ = self.token_embedding(F_token[:, :-1])
        F_token_emb = self.dropout(F_token_emb_ + F_pos_emb)


        G_pos_emb = self.pos_embedding(
            torch.arange(0, G_token.shape[1] - 1)
            .unsqueeze(0)
            .repeat(G_token.shape[0], 1)
            .type_as(G_token)
        )
        G_token_emb_ = self.token_embedding(G_token[:, :-1])
        G_token_emb = self.dropout(G_token_emb_ + G_pos_emb)


        F_padding_mask, F_token_mask = self.make_trg_mask(F_token[:, :-1])
        G_padding_mask, G_token_mask = self.make_trg_mask(G_token[:, :-1])

        dec_F_output = self.F_decoder_transformer(
            F_token_emb.permute(1, 0, 2),
            enc_output.permute(1, 0, 2),
            F_token_mask.bool(),
            tgt_key_padding_mask=F_padding_mask.bool()
        )

        dec_G_output = self.G_decoder_transformer(
            G_token_emb.permute(1, 0, 2),
            enc_output.permute(1, 0, 2),
            G_token_mask.bool(),
            tgt_key_padding_mask=G_padding_mask.bool()
        )

        F_output = self.fc_out(dec_F_output)

        G_output = self.fc_out(dec_G_output)
        
        return F_output, G_output, F_token, G_token

    def compute_loss(self, F_output, G_output, F_token, G_token):
        
        F_output = F_output.permute(1, 0, 2).contiguous().view(-1, F_output.shape[-1])
        F_token = F_token[:, 1:].contiguous().view(-1)
        F_loss = self.criterion(F_output, F_token)

        G_output = G_output.permute(1, 0, 2).contiguous().view(-1, G_output.shape[-1])
        
        G_token = G_token[:, 1:].contiguous().view(-1)


        

        G_loss = self.criterion(G_output, G_token)
        return F_loss + G_loss

    def training_step(self, batch, _):
       
        F_output, G_output, F_token, G_token = self.forward(batch)
        loss = self.compute_loss(F_output, G_output, F_token, G_token)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, on_step=False)
        return loss

    def validation_step(self, batch, _):

        F_output, G_output, F_token, G_token = self.forward(batch)
        loss = self.compute_loss(F_output, G_output, F_token, G_token)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
        return optimizer
    
    def fitfunc_F_trans(self, node_state, y, dimension, cfg_params):
        beam_size = cfg_params[0]
        cfg = cfg_params[1]
        env=cfg_params[2]

        node_state = torch.tensor(node_state, device=self.device)


        y = torch.tensor(y, device=self.device)
        with torch.no_grad():
            node_state_emb = self.node_state_embedding(node_state)
            y_emb = self.y_embedding(y)
            enc_emb = node_state_emb + y_emb
            enc_output = self.encode_set_transformer(enc_emb)
            src_enc = enc_output
            shape_enc_src = (beam_size,) + src_enc.shape[1:]
            enc_output = src_enc.unsqueeze(1).expand((1, beam_size) + src_enc.shape[1:]).contiguous().view(
                shape_enc_src)
            assert enc_output.size(0) == beam_size
            F_generated = torch.zeros(
                [beam_size, self.cfg.length_eq],
                dtype=torch.long,
                device=self.device,
            )
            F_generated[:, 0] = 1
            F_beam_scores = torch.zeros(beam_size, device=self.device, dtype=torch.long)
            F_beam_scores[1:] = -1e9
            F_cur_len = torch.tensor(1, device=self.device, dtype=torch.int64)
            beam_scorer = transformers.generation.BeamSearchScorer(
                batch_size=1,
                num_beams=beam_size,
                device=node_state_emb.device,
                length_penalty=1,
                do_early_stopping=False,
                num_beam_hyps_to_keep=1,
                max_length=self.cfg.length_eq,
            )

            while F_cur_len < self.cfg.length_eq:
                F_padding_mask, F_token_mask = self.make_trg_mask(F_generated[:, :F_cur_len])
                F_pos_emb = self.pos_embedding(
                    torch.arange(0, F_cur_len)
                    .unsqueeze(0)
                    .repeat(F_generated.shape[0], 1)
                    .type_as(F_generated)
                )
                F_token_emb_ = self.token_embedding(F_generated[:, :F_cur_len])
                F_token_emb = self.dropout(F_token_emb_ + F_pos_emb)
                dec_F_output = self.F_decoder_transformer(
                    F_token_emb.permute(1, 0, 2),
                    enc_output.permute(1, 0, 2),
                    F_token_mask.bool(),
                    tgt_key_padding_mask=F_padding_mask.bool()
                )
                F_output = self.fc_out(dec_F_output)
                F_output = F_output.permute(1, 0, 2).contiguous()

                F_scores = F.log_softmax(F_output[:, -1:, :], dim=-1).squeeze(1)
                
                F_n_words = F_scores.shape[-1]
                _F_scores = F_scores + F_beam_scores[:, None].expand_as(F_scores)

                _F_scores = _F_scores.view(1,beam_size * F_n_words)

                F_next_scores, F_next_words = torch.topk(_F_scores, 2 * beam_size, dim=1, largest=True, sorted=True)
                next_indices = torch.div(F_next_words, F_n_words, rounding_mode="floor")
                next_tokens = F_next_words % F_n_words

                beam_outputs = beam_scorer.process(
                    F_generated[:,:F_cur_len],
                    F_next_scores,
                    next_tokens,
                    next_indices,
                    pad_token_id=torch.Tensor([0]),
                    eos_token_id=torch.Tensor([2]),
                    beam_indices=None,
                    decoder_prompt_len=1,
                )

                F_beam_scores = beam_outputs["next_beam_scores"]
                F_beam_words = beam_outputs["next_beam_tokens"]
                F_beam_idx = beam_outputs["next_beam_indices"]


                F_generated = F_generated[F_beam_idx, :]
                F_generated[:, F_cur_len] = F_beam_words

                F_cur_len = F_cur_len + torch.tensor(
                    1, device=self.device, dtype=torch.int64
                )
            
            L_bfgs = []
            P_bfgs = []
            ys=[]
            vals=[]

            

            F_beam_select=[F_ww for __, F_ww,_    in sorted(beam_scorer._beam_hyps[0].beams, key=lambda x: x[0], reverse=True)]



            for F_ww in F_beam_select:
                pred_w_c, loss_bfgs,_y,val=bfgs_F(F_ww,node_state,y,dimension,cfg,env)
                if pred_w_c is None:
                    continue
                P_bfgs.append(pred_w_c)
                L_bfgs.append(loss_bfgs)
                ys.append(_y)
                vals.append(val)
            if all(np.isnan(np.array(L_bfgs))):
                print("Warning all nans")
                L_bfgs = float("nan")
                best_L_bfgs = "wrong"
                metrics_output="wrong"
                best_preds_bfgs='wrong'
            else:
                best_preds_bfgs=P_bfgs[np.nanargmin(L_bfgs)]
                best_L_bfgs=np.nanmin(L_bfgs)

                _y=ys[np.nanargmin(L_bfgs)]
                val=vals[np.nanargmin(L_bfgs)]

                
                metrics_output=cal_metrics(_y,val)
                
            output = {'all_bfgs_preds':P_bfgs, 
                      'all_bfgs_loss':L_bfgs, 
                      'best_bfgs_preds':best_preds_bfgs, 
                      'best_bfgs_loss':best_L_bfgs}
            return output,metrics_output
        
    def fitfunc_FG(self, node_state, neighbour_state,info, y, dimension,cfg_params):
        beam_size = cfg_params[0]
        cfg = cfg_params[1]
        env=cfg_params[2]

        node_state = torch.tensor(node_state, device=self.device)
        neighbour_state = torch.tensor(neighbour_state, device=self.device)



        info = torch.tensor(info, device=self.device)


        y = torch.tensor(y, device=self.device)
        with torch.no_grad():
            node_state_emb = self.node_state_embedding(node_state)
            neighbour_state_emb = self.neighbour_state_embedding(node_state_emb,neighbour_state, info)
            y_emb = self.y_embedding(y)

            enc_emb = node_state_emb + neighbour_state_emb + y_emb

            enc_output = self.encode_set_transformer(enc_emb)



            src_enc = enc_output
            shape_enc_src = (beam_size,) + src_enc.shape[1:]

            enc_output = src_enc.unsqueeze(1).expand((1, beam_size) + src_enc.shape[1:]).contiguous().view(
                shape_enc_src)
            assert enc_output.size(0) == beam_size

            F_generated = torch.zeros(
                [beam_size, self.cfg.length_eq],
                dtype=torch.long,
                device=self.device,
            )

            G_generated = torch.zeros(
                [beam_size, self.cfg.length_eq],
                dtype=torch.long,
                device=self.device,
            )

            F_generated[:, 0] = 1
            G_generated[:, 0] = 1

            F_generated_hyps = BeamHypotheses(beam_size, self.cfg.length_eq, 1.0, 1)
            G_generated_hyps = BeamHypotheses(beam_size, self.cfg.length_eq, 1.0, 1)

            F_beam_scores = torch.zeros(beam_size, device=self.device, dtype=torch.long)
            F_beam_scores[1:] = -1e9
            F_cur_len = torch.tensor(1, device=self.device, dtype=torch.int64)

            while F_cur_len < self.cfg.length_eq:
                F_padding_mask, F_token_mask = self.make_trg_mask(F_generated[:, :F_cur_len])
                F_pos_emb = self.pos_embedding(
                    torch.arange(0, F_cur_len)
                    .unsqueeze(0)
                    .repeat(F_generated.shape[0], 1)
                    .type_as(F_generated)
                )
                F_token_emb_ = self.token_embedding(F_generated[:, :F_cur_len])
                F_token_emb = self.dropout(F_token_emb_ + F_pos_emb)
                dec_F_output = self.F_decoder_transformer(
                    F_token_emb.permute(1, 0, 2),
                    enc_output.permute(1, 0, 2),
                    F_token_mask.bool(),
                    tgt_key_padding_mask=F_padding_mask.bool()
                )
                F_output = self.fc_out(dec_F_output)
                F_output = F_output.permute(1, 0, 2).contiguous()
                F_scores = F.log_softmax(F_output[:, -1:, :], dim=-1).squeeze(1)
                F_n_words = F_scores.shape[-1]
                _F_scores = F_scores + F_beam_scores[:, None].expand_as(F_scores)
                _F_scores = _F_scores.view(beam_size * F_n_words)

                F_next_scores, F_next_words = torch.topk(_F_scores, 2 * beam_size, dim=0, largest=True, sorted=True)
                assert len(F_next_scores) == len(F_next_words) == 2 * beam_size

                F_next_sent_beam = []
                for F_idx, F_value in zip(F_next_words, F_next_scores):
                    F_beam_id = F_idx // F_n_words
                    F_word_id = F_idx % F_n_words

                    if F_word_id == cfg.word2id["F"] or F_cur_len + 1 == self.cfg.length_eq:
                        F_generated_hyps.add(
                            F_generated[F_beam_id, :F_cur_len].clone().cpu(),
                            F_value.item(),
                        )

                    else:
                        F_next_sent_beam.append((F_value, F_word_id, F_beam_id))

                    if len(F_next_sent_beam) == beam_size:
                        break

                assert (
                    len(F_next_sent_beam) == 0
                    if F_cur_len + 1 == self.cfg.length_eq
                    else beam_size
                )

                if len(F_next_sent_beam) == 0:
                    F_next_sent_beam = [(0, self.trg_pad_idx, 0)] * beam_size
                
                assert len(F_next_sent_beam) == beam_size

                F_beam_scores = torch.tensor([x[0] for x in F_next_sent_beam], device=self.device)

                F_beam_words = torch.tensor([x[1] for x in F_next_sent_beam], device=self.device)

                F_beam_idx = torch.tensor([x[2] for x in F_next_sent_beam], device=self.device)

                F_generated = F_generated[F_beam_idx, :]
                F_generated[:, F_cur_len] = F_beam_words

                F_cur_len = F_cur_len + torch.tensor(
                    1, device=self.device, dtype=torch.int64
                )

            G_beam_scores = torch.zeros(beam_size, device=self.device, dtype=torch.long)
            G_beam_scores[1:] = -1e9
            G_cur_len = torch.tensor(1, device=self.device, dtype=torch.int64)

            while G_cur_len < self.cfg.length_eq:
                G_padding_mask, G_token_mask = self.make_trg_mask(G_generated[:, :G_cur_len])

                G_pos_emb = self.pos_embedding(
                    torch.arange(0, G_cur_len)
                    .unsqueeze(0)
                    .repeat(G_generated.shape[0], 1)
                    .type_as(G_generated)
                )
                G_token_emb_ = self.token_embedding(G_generated[:, :G_cur_len])
                G_token_emb = self.dropout(G_token_emb_ + G_pos_emb)

                dec_G_output = self.G_decoder_transformer(
                    G_token_emb.permute(1, 0, 2),
                    enc_output.permute(1, 0, 2),
                    G_token_mask.bool(),
                    tgt_key_padding_mask=G_padding_mask.bool()
                )
                G_output = self.fc_out(dec_G_output)
                G_output = G_output.permute(1, 0, 2).contiguous()
                G_scores = F.log_softmax(G_output[:, -1:, :], dim=-1).squeeze(1)
                G_n_words = G_scores.shape[-1]
                _G_scores = G_scores + G_beam_scores[:, None].expand_as(G_scores)
                _G_scores = _G_scores.view(beam_size * G_n_words)
                G_next_scores, G_next_words = torch.topk(_G_scores, 2 * beam_size, dim=0, largest=True, sorted=True)
                assert len(G_next_scores) == len(G_next_words) == 2 * beam_size
                G_next_sent_beam = []

                for G_idx, G_value in zip(G_next_words, G_next_scores):
                    G_beam_id = G_idx // G_n_words
                    G_word_id = G_idx % G_n_words

                    if (
                            G_word_id == cfg.word2id["F"] or G_cur_len + 1 == self.cfg.length_eq
                    ):
                        G_generated_hyps.add(
                            G_generated[G_beam_id, :G_cur_len].clone().cpu(),
                            G_value.item(),
                        )

                    else:
                        G_next_sent_beam.append((G_value, G_word_id, G_beam_id))

                    if len(G_next_sent_beam) == beam_size:
                        break
                assert (
                    len(G_next_sent_beam) == 0
                    if G_cur_len + 1 == self.cfg.length_eq
                    else beam_size
                )

                if len(G_next_sent_beam) == 0:
                    G_next_sent_beam = [(0, self.trg_pad_idx, 0)] * beam_size

                assert len(G_next_sent_beam) == beam_size

                G_beam_scores = torch.tensor([x[0] for x in G_next_sent_beam], device=self.device)

                G_beam_words = torch.tensor([x[1] for x in G_next_sent_beam], device=self.device)

                G_beam_idx = torch.tensor([x[2] for x in G_next_sent_beam], device=self.device)

                G_generated = G_generated[G_beam_idx, :]
                G_generated[:, G_cur_len] = G_beam_words

                G_cur_len = G_cur_len + torch.tensor(
                    1, device=self.device, dtype=torch.int64
                )

            best_preds_bfgs = []
            best_L_bfgs = []
            L_bfgs = []
            P_bfgs = []

            ys=[]
            vals=[]


            F_beam_select=[F_ww for __, F_ww in sorted(F_generated_hyps.hyp, key=lambda x: x[0], reverse=True)]
            G_beam_select=[G_ww for __, G_ww in sorted(G_generated_hyps.hyp, key=lambda x: x[0], reverse=True)]


            for F_ww,G_ww in zip(F_beam_select,G_beam_select):
                pred_w_c, loss_bfgs,_y,val=bfgs_FG(F_ww,G_ww,node_state,neighbour_state,info,y,dimension,cfg,env)

                if pred_w_c is None:
                    continue
                P_bfgs.append(pred_w_c)
                L_bfgs.append(loss_bfgs)
                ys.append(_y)
                vals.append(val)
                
            if all(np.isnan(np.array(L_bfgs))):
                print("Warning all nans")
                L_bfgs = float("nan")
                best_L_bfgs = "wrong"
                metrics_output="wrong"
                best_preds_bfgs='wrong'
            else:
                best_preds_bfgs=P_bfgs[np.nanargmin(L_bfgs)]
                best_L_bfgs=np.nanmin(L_bfgs)
                _y=ys[np.nanargmin(L_bfgs)]
                val=vals[np.nanargmin(L_bfgs)]
                metrics_output=cal_metrics(_y,val)

            output = {'all_bfgs_preds':P_bfgs, 
                    'all_bfgs_loss':L_bfgs, 
                    'best_bfgs_preds':best_preds_bfgs, 
                    'best_bfgs_loss':best_L_bfgs}
            return output,metrics_output
            

    def fitfunc_FG_Dy(self, node_state, neighbour_state, info, y, dimension,cfg_params):
        beam_size = cfg_params[0]
        cfg = cfg_params[1]
        env=cfg_params[2]

        node_state = torch.tensor(node_state, device=self.device)
        neighbour_state = torch.tensor(neighbour_state, device=self.device)
        info = torch.tensor(info, device=self.device)
        y = torch.tensor(y, device=self.device)
        with torch.no_grad():
            node_state_emb = self.node_state_embedding(node_state)
            neighbour_state_emb = self.neighbour_state_embedding(node_state_emb,neighbour_state, info)
            y_emb = self.y_embedding(y)

            enc_emb = node_state_emb + neighbour_state_emb + y_emb
            enc_output = self.encode_set_transformer(enc_emb)
            src_enc = enc_output
            shape_enc_src = (beam_size,) + src_enc.shape[1:]
            enc_output = src_enc.unsqueeze(1).expand((1, beam_size) + src_enc.shape[1:]).contiguous().view(
                shape_enc_src)
            assert enc_output.size(0) == beam_size

            F_generated = torch.zeros(
                [beam_size, self.cfg.length_eq],
                dtype=torch.long,
                device=self.device,
            )

            G_generated = torch.zeros(
                [beam_size, self.cfg.length_eq],
                dtype=torch.long,
                device=self.device,
            )
            F_generated[:, 0] = 1
            G_generated[:, 0] = 1

            F_generated_hyps = BeamHypotheses(beam_size, self.cfg.length_eq, 1.0, 1)
            G_generated_hyps = BeamHypotheses(beam_size, self.cfg.length_eq, 1.0, 1)

            F_beam_scores = torch.zeros(beam_size, device=self.device, dtype=torch.long)
            F_beam_scores[1:] = -1e9
            F_cur_len = torch.tensor(1, device=self.device, dtype=torch.int64)

            while F_cur_len < self.cfg.length_eq:
                F_padding_mask, F_token_mask = self.make_trg_mask(F_generated[:, :F_cur_len])
                F_pos_emb = self.pos_embedding(
                    torch.arange(0, F_cur_len)
                    .unsqueeze(0)
                    .repeat(F_generated.shape[0], 1)
                    .type_as(F_generated)
                )
                F_token_emb_ = self.token_embedding(F_generated[:, :F_cur_len])
                F_token_emb = self.dropout(F_token_emb_ + F_pos_emb)

 

                dec_F_output = self.F_decoder_transformer(
                    F_token_emb.permute(1, 0, 2),
                    enc_output.permute(1, 0, 2),
                    F_token_mask.bool(),
                    tgt_key_padding_mask=F_padding_mask.bool()
                )
                
                F_output = self.fc_out(dec_F_output)
                
                F_output = F_output.permute(1, 0, 2).contiguous()
                F_scores = F.log_softmax(F_output[:, -1:, :], dim=-1).squeeze(1)
                F_n_words = F_scores.shape[-1]
                _F_scores = F_scores + F_beam_scores[:, None].expand_as(F_scores)
                _F_scores = _F_scores.view(beam_size * F_n_words)

                F_next_scores, F_next_words = torch.topk(_F_scores, 2 * beam_size, dim=0, largest=True, sorted=True)
                assert len(F_next_scores) == len(F_next_words) == 2 * beam_size

                F_next_sent_beam = []
                for F_idx, F_value in zip(F_next_words, F_next_scores):
                    F_beam_id = F_idx // F_n_words
                    F_word_id = F_idx % F_n_words

                    if F_word_id == cfg.word2id["F"] or F_cur_len + 1 == self.cfg.length_eq:
                        F_generated_hyps.add(
                            F_generated[F_beam_id, :F_cur_len].clone().cpu(),
                            F_value.item(),
                        )

                    else:
                        F_next_sent_beam.append((F_value, F_word_id, F_beam_id))

                    if len(F_next_sent_beam) == beam_size:
                        break

                assert (
                    len(F_next_sent_beam) == 0
                    if F_cur_len + 1 == self.cfg.length_eq
                    else beam_size
                )

                if len(F_next_sent_beam) == 0:
                    F_next_sent_beam = [(0, self.trg_pad_idx, 0)] * beam_size
                
                assert len(F_next_sent_beam) == beam_size

                F_beam_scores = torch.tensor([x[0] for x in F_next_sent_beam], device=self.device)

                F_beam_words = torch.tensor([x[1] for x in F_next_sent_beam], device=self.device)

                F_beam_idx = torch.tensor([x[2] for x in F_next_sent_beam], device=self.device)

                F_generated = F_generated[F_beam_idx, :]
                F_generated[:, F_cur_len] = F_beam_words

                F_cur_len = F_cur_len + torch.tensor(
                    1, device=self.device, dtype=torch.int64
                )

            G_beam_scores = torch.zeros(beam_size, device=self.device, dtype=torch.long)
            G_beam_scores[1:] = -1e9
            G_cur_len = torch.tensor(1, device=self.device, dtype=torch.int64)

            while G_cur_len < self.cfg.length_eq:
                G_padding_mask, G_token_mask = self.make_trg_mask(G_generated[:, :G_cur_len])

                G_pos_emb = self.pos_embedding(
                    torch.arange(0, G_cur_len)
                    .unsqueeze(0)
                    .repeat(G_generated.shape[0], 1)
                    .type_as(G_generated)
                )
                G_token_emb_ = self.token_embedding(G_generated[:, :G_cur_len])
                G_token_emb = self.dropout(G_token_emb_ + G_pos_emb)

                dec_G_output = self.G_decoder_transformer(
                    G_token_emb.permute(1, 0, 2),
                    enc_output.permute(1, 0, 2),
                    G_token_mask.bool(),
                    tgt_key_padding_mask=G_padding_mask.bool()
                )
                G_output = self.fc_out(dec_G_output)
                G_output = G_output.permute(1, 0, 2).contiguous()
                G_scores = F.log_softmax(G_output[:, -1:, :], dim=-1).squeeze(1)
                G_n_words = G_scores.shape[-1]
                _G_scores = G_scores + G_beam_scores[:, None].expand_as(G_scores)
                _G_scores = _G_scores.view(beam_size * G_n_words)
                G_next_scores, G_next_words = torch.topk(_G_scores, 2 * beam_size, dim=0, largest=True, sorted=True)
                assert len(G_next_scores) == len(G_next_words) == 2 * beam_size
                G_next_sent_beam = []

                for G_idx, G_value in zip(G_next_words, G_next_scores):
                    G_beam_id = G_idx // G_n_words
                    G_word_id = G_idx % G_n_words

                    if (
                            G_word_id == cfg.word2id["F"] or G_cur_len + 1 == self.cfg.length_eq
                    ):
                        G_generated_hyps.add(
                            G_generated[G_beam_id, :G_cur_len].clone().cpu(),
                            G_value.item(),
                        )

                    else:
                        G_next_sent_beam.append((G_value, G_word_id, G_beam_id))

                    if len(G_next_sent_beam) == beam_size:
                        break
                assert (
                    len(G_next_sent_beam) == 0
                    if G_cur_len + 1 == self.cfg.length_eq
                    else beam_size
                )

                if len(G_next_sent_beam) == 0:
                    G_next_sent_beam = [(0, self.trg_pad_idx, 0)] * beam_size

                assert len(G_next_sent_beam) == beam_size

                G_beam_scores = torch.tensor([x[0] for x in G_next_sent_beam], device=self.device)

                G_beam_words = torch.tensor([x[1] for x in G_next_sent_beam], device=self.device)

                G_beam_idx = torch.tensor([x[2] for x in G_next_sent_beam], device=self.device)

                G_generated = G_generated[G_beam_idx, :]
                G_generated[:, G_cur_len] = G_beam_words

                G_cur_len = G_cur_len + torch.tensor(
                    1, device=self.device, dtype=torch.int64
                )

            best_preds_bfgs = []
            best_L_bfgs = []
            L_bfgs = []
            P_bfgs = []

            ys=[]
            vals=[]


            F_beam_select=[F_ww for __, F_ww in sorted(F_generated_hyps.hyp, key=lambda x: x[0], reverse=True)]
            G_beam_select=[G_ww for __, G_ww in sorted(G_generated_hyps.hyp, key=lambda x: x[0], reverse=True)]

            for F_ww,G_ww in zip(F_beam_select,G_beam_select):
                pred_w_c, loss_bfgs,_y,val=bfgs_FG(F_ww,G_ww,node_state,neighbour_state,info,y,dimension,cfg,env)
                if pred_w_c is None:
                    continue
                P_bfgs.append(pred_w_c)
                L_bfgs.append(loss_bfgs)
                ys.append(_y)
                vals.append(val)
                
            if all(np.isnan(np.array(L_bfgs))):
                print("Warning all nans")
                L_bfgs = float("nan")
                best_L_bfgs = "wrong"
                metrics_output="wrong"
                best_preds_bfgs='wrong'
            else:
                best_preds_bfgs=P_bfgs[np.nanargmin(L_bfgs)]
                best_L_bfgs=np.nanmin(L_bfgs)
                _y=ys[np.nanargmin(L_bfgs)]
                val=vals[np.nanargmin(L_bfgs)]
                metrics_output=cal_metrics(_y,val)

            output = {'all_bfgs_preds':P_bfgs, 
                      'all_bfgs_loss':L_bfgs, 
                      'best_bfgs_preds':best_preds_bfgs, 
                      'best_bfgs_loss':best_L_bfgs}
            return output,metrics_output
    

    def fitfunc_FG_Dy_real(self, node_state, neighbour_state, info, y, dimension,cfg_params):
        beam_size = cfg_params[0]
        cfg = cfg_params[1]
        env=cfg_params[2]

        node_state = torch.tensor(node_state, device=self.device)
        neighbour_state = torch.tensor(neighbour_state, device=self.device)

        info = torch.tensor(info, device=self.device)


        y = torch.tensor(y, device=self.device)
        with torch.no_grad():
            node_state_emb = self.node_state_embedding(node_state)
            neighbour_state_emb = self.neighbour_state_embedding(node_state_emb,neighbour_state, info)
            y_emb = self.y_embedding(y)

            enc_emb = node_state_emb + neighbour_state_emb + y_emb

            enc_output = self.encode_set_transformer(enc_emb)

            src_enc = enc_output
            shape_enc_src = (beam_size,) + src_enc.shape[1:]

            enc_output = src_enc.unsqueeze(1).expand((1, beam_size) + src_enc.shape[1:]).contiguous().view(
                shape_enc_src)
            assert enc_output.size(0) == beam_size

            F_generated = torch.zeros(
                [beam_size, self.cfg.length_eq],
                dtype=torch.long,
                device=self.device,
            )

            G_generated = torch.zeros(
                [beam_size, self.cfg.length_eq],
                dtype=torch.long,
                device=self.device,
            )
            
            F_generated[:, 0] = 1
            G_generated[:, 0] = 1

            F_generated_hyps = BeamHypotheses(beam_size, self.cfg.length_eq, 1.0, 1)
            G_generated_hyps = BeamHypotheses(beam_size, self.cfg.length_eq, 1.0, 1)

            F_beam_scores = torch.zeros(beam_size, device=self.device, dtype=torch.long)
            F_beam_scores[1:] = -1e9
            F_cur_len = torch.tensor(1, device=self.device, dtype=torch.int64)

            while F_cur_len < self.cfg.length_eq:
                F_padding_mask, F_token_mask = self.make_trg_mask(F_generated[:, :F_cur_len])
                F_pos_emb = self.pos_embedding(
                    torch.arange(0, F_cur_len)
                    .unsqueeze(0)
                    .repeat(F_generated.shape[0], 1)
                    .type_as(F_generated)
                )
                F_token_emb_ = self.token_embedding(F_generated[:, :F_cur_len])
                F_token_emb = self.dropout(F_token_emb_ + F_pos_emb)

 

                dec_F_output = self.F_decoder_transformer(
                    F_token_emb.permute(1, 0, 2),
                    enc_output.permute(1, 0, 2),
                    F_token_mask.bool(),
                    tgt_key_padding_mask=F_padding_mask.bool()
                )
                
                F_output = self.fc_out(dec_F_output)
                
                F_output = F_output.permute(1, 0, 2).contiguous()
                F_scores = F.log_softmax(F_output[:, -1:, :], dim=-1).squeeze(1)
                F_n_words = F_scores.shape[-1]
                _F_scores = F_scores + F_beam_scores[:, None].expand_as(F_scores)
                _F_scores = _F_scores.view(beam_size * F_n_words)

                F_next_scores, F_next_words = torch.topk(_F_scores, 2 * beam_size, dim=0, largest=True, sorted=True)
                assert len(F_next_scores) == len(F_next_words) == 2 * beam_size

                F_next_sent_beam = []
                for F_idx, F_value in zip(F_next_words, F_next_scores):
                    F_beam_id = F_idx // F_n_words
                    F_word_id = F_idx % F_n_words

                    if F_word_id == cfg.word2id["F"] or F_cur_len + 1 == self.cfg.length_eq:
                        F_generated_hyps.add(
                            F_generated[F_beam_id, :F_cur_len].clone().cpu(),
                            F_value.item(),
                        )

                    else:
                        F_next_sent_beam.append((F_value, F_word_id, F_beam_id))

                    if len(F_next_sent_beam) == beam_size:
                        break

                assert (
                    len(F_next_sent_beam) == 0
                    if F_cur_len + 1 == self.cfg.length_eq
                    else beam_size
                )

                if len(F_next_sent_beam) == 0:
                    F_next_sent_beam = [(0, self.trg_pad_idx, 0)] * beam_size
                
                assert len(F_next_sent_beam) == beam_size

                F_beam_scores = torch.tensor([x[0] for x in F_next_sent_beam], device=self.device)

                F_beam_words = torch.tensor([x[1] for x in F_next_sent_beam], device=self.device)

                F_beam_idx = torch.tensor([x[2] for x in F_next_sent_beam], device=self.device)

                F_generated = F_generated[F_beam_idx, :]
                F_generated[:, F_cur_len] = F_beam_words

                F_cur_len = F_cur_len + torch.tensor(
                    1, device=self.device, dtype=torch.int64
                )

            G_beam_scores = torch.zeros(beam_size, device=self.device, dtype=torch.long)
            G_beam_scores[1:] = -1e9
            G_cur_len = torch.tensor(1, device=self.device, dtype=torch.int64)

            while G_cur_len < self.cfg.length_eq:
                G_padding_mask, G_token_mask = self.make_trg_mask(G_generated[:, :G_cur_len])

                G_pos_emb = self.pos_embedding(
                    torch.arange(0, G_cur_len)
                    .unsqueeze(0)
                    .repeat(G_generated.shape[0], 1)
                    .type_as(G_generated)
                )
                G_token_emb_ = self.token_embedding(G_generated[:, :G_cur_len])
                G_token_emb = self.dropout(G_token_emb_ + G_pos_emb)

                dec_G_output = self.G_decoder_transformer(
                    G_token_emb.permute(1, 0, 2),
                    enc_output.permute(1, 0, 2),
                    G_token_mask.bool(),
                    tgt_key_padding_mask=G_padding_mask.bool()
                )
                G_output = self.fc_out(dec_G_output)
                G_output = G_output.permute(1, 0, 2).contiguous()
                G_scores = F.log_softmax(G_output[:, -1:, :], dim=-1).squeeze(1)
                G_n_words = G_scores.shape[-1]
                _G_scores = G_scores + G_beam_scores[:, None].expand_as(G_scores)
                _G_scores = _G_scores.view(beam_size * G_n_words)
                G_next_scores, G_next_words = torch.topk(_G_scores, 2 * beam_size, dim=0, largest=True, sorted=True)
                assert len(G_next_scores) == len(G_next_words) == 2 * beam_size
                G_next_sent_beam = []

                for G_idx, G_value in zip(G_next_words, G_next_scores):
                    G_beam_id = G_idx // G_n_words
                    G_word_id = G_idx % G_n_words

                    if (
                            G_word_id == cfg.word2id["F"] or G_cur_len + 1 == self.cfg.length_eq
                    ):
                        G_generated_hyps.add(
                            G_generated[G_beam_id, :G_cur_len].clone().cpu(),
                            G_value.item(),
                        )

                    else:
                        G_next_sent_beam.append((G_value, G_word_id, G_beam_id))

                    if len(G_next_sent_beam) == beam_size:
                        break
                assert (
                    len(G_next_sent_beam) == 0
                    if G_cur_len + 1 == self.cfg.length_eq
                    else beam_size
                )

                if len(G_next_sent_beam) == 0:
                    G_next_sent_beam = [(0, self.trg_pad_idx, 0)] * beam_size

                assert len(G_next_sent_beam) == beam_size

                G_beam_scores = torch.tensor([x[0] for x in G_next_sent_beam], device=self.device)

                G_beam_words = torch.tensor([x[1] for x in G_next_sent_beam], device=self.device)

                G_beam_idx = torch.tensor([x[2] for x in G_next_sent_beam], device=self.device)

                G_generated = G_generated[G_beam_idx, :]
                G_generated[:, G_cur_len] = G_beam_words

                G_cur_len = G_cur_len + torch.tensor(
                    1, device=self.device, dtype=torch.int64
                )

            F_beam_select=[F_ww for __, F_ww in sorted(F_generated_hyps.hyp, key=lambda x: x[0], reverse=True)]
            G_beam_select=[G_ww for __, G_ww in sorted(G_generated_hyps.hyp, key=lambda x: x[0], reverse=True)]
        return F_beam_select,G_beam_select