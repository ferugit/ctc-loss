import torch


class CTCLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def compute_log_alpha(self, x, y, blank):
        
        # As wrap-ctc
        x = torch.nn.functional.log_softmax(x, dim=1)
        
        T = len(x)
        U = len(y)
        S = 2*U + 1
        z = [] # [_, y1, _, y2, _, y3, _]
        
        # Get the sequence staating and ending with blank
        # and using a blank between every token
        for i in range(S):
            label = blank if (i+1) % 2 else y[int(i/2)].item()
            z.append(label)
        
        log_alphas = []

        for t in range(T):
            eps = 1e-30
            log_alpha_t = torch.log(torch.zeros(S) + eps) # w/o eps, gradients will be nan
            
            # First time step
            if t == 0:
                # Initialize with blank or fisrt token
                log_alpha_t[0] = x[0, blank]
                log_alpha_t[1] = x[0, z[1]]
            
            else:
                # Get last transition prob
                log_alpha_t_1 = log_alphas[-1]
                
                for s in range(S):
                    
                    if s == 0:
                        log_alpha_t[s] = log_alpha_t_1[s] + x[t, z[s]]
                        
                    if s == 1:
                        log_alpha_t[s] = torch.logsumexp(log_alpha_t_1[s-1:s+1], dim=0) + x[t, z[s]]
                        
                    if s > 1:
                        # Case 1 described here https://distill.pub/2017/ctc/                 
                        if z[s] == blank or z[s-2] == z[s]:
                            log_alpha_t[s] = torch.logsumexp(log_alpha_t_1[s-1:s+1], dim=0) + x[t, z[s]]
                        # Case 2 described here https://distill.pub/2017/ctc/
                        # Skip previous token
                        else:
                            log_alpha_t[s] = torch.logsumexp(log_alpha_t_1[s-2:s+1], dim=0) + x[t, z[s]]
            #print("")
            #print(log_alpha_t)
            log_alphas.append(log_alpha_t)
            
        return torch.stack(log_alphas)

    def forward(self, log_probs, targets, input_lengths,
     target_lengths, reduction='none', blank=0):
        """
        Compute CTC loss

        Parameters
        ----------
        log_probs:
            FloatTensor with log probabilies, shape
             [max(input_lengths), batch_size, n_tokens]
        targets:
            LongTensor, shape 
            [batch_size, max(target_lengths)]
        input_lengths:
            LongTensor, shape [batch_size]
        target_lengths:
            LongTensor, shape [batch_size]
        reduction:
            Type of reduction to apply to loss
            'none' or 'avg'
        blank:
            Token used to represent the blank

        Returns
        -------
        losses:
            CTC loss.
        """

        # Get batch size
        bs = len(input_lengths)

        # Initilize losses
        losses = []

        # Iterate log probs
        for i in range(bs):
            x = log_probs[:input_lengths[i], i, :]
            y = targets[i, :target_lengths[i]]
            log_alpha = self.compute_log_alpha(x, y, blank) # [T, 2U + 1]
            loss = -torch.logsumexp(log_alpha[-1, -2:], dim=0)
            losses.append(loss)
        
        # Return loss
        losses = torch.stack(losses)
        return losses.mean() if(reduction == 'avg') else losses