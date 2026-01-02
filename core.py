import numpy as np
import cupy as cp

class Tensor(cp.ndarray): # For typehints
    pass

class SoftmaxCrossEntropy:
    def __init__(self):
        self.logits: Tensor | None = None
        self.targets: Tensor | None = None
        self.probs: Tensor | None = None
        
    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        self.logits = logits
        self.targets = targets
        # x is (B, seq_len, out_dim) (logits)
        # targets is one hot encoded of shape (B, seq_len, out_dim) so for each token we have correct next one, as one-hot
        max_logits = cp.max(logits, axis=-1, keepdims=True) # (B, seq_len, 1)
        shifted_logits = logits - max_logits # Broadcasts, (B, seq_len, out_dim) 
        
        exp_logits = cp.exp(shifted_logits) # (B, seq_len, out_dim)
        exp_sum = cp.sum(exp_logits, axis=-1, keepdims=True) # (B, seq_len, 1)
        self.probs = exp_logits / exp_sum # Broadcasts, (B, seq_len, out_dim)
        
        log_probs = cp.log(self.probs) # takes the natural log to calc the neg log likelihood
        log_probs = log_probs * targets # plucks out correct class via one-hot
        batch_loss = -cp.sum(log_probs, axis=-1) # (B, seq_len)
        batch_loss = cp.mean(batch_loss)
        
        return batch_loss
    
    def backward(self) -> Tensor:
        batch_size = self.logits.shape[0]
        
        grad = (self.probs - self.targets) / batch_size
        
        return grad
    
class MSELoss:
    def __init__(self):
        self.preds: Tensor | None = None
        self.targets: Tensor | None = None
        
    def forward(self, preds, targets):
        # x is (B, seq_len, out_dim)
        # preds is our logits in this case
        self.preds = preds
        self.targets = targets
        
        loss = (targets - preds) ** 2
        loss = cp.mean(loss)

        return loss
    
    def backward(self):
        # we want to get dL/dpreds
        # we just do 2/N(pred-targer)
        
        grad = 2 / self.preds.shape[0] * (self.preds - self.targets)

        # It is important to notice the sign and diff the power of 2. 
        # The loss has to know whether prediction is above or below loss
        
        return grad 
    
class Embedding:
    def __init__(self, input_dim, embed_dim):
        self.input_cache: Tensor | None = None
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        self.embeddings: Tensor = cp.random.randn(input_dim, embed_dim)
        
    def forward(self, x) -> Tensor:
        self.input_cache = x
        
        return self.embeddings[x] # (x.shape, embed_dim)
    
    def backward(self, out_grad):
        self.dembeddings = cp.zeros_like(self.embeddings)
        cp.add.at(self.dembeddings, self.input_cache, out_grad)
        
        # For each input_cache[a, b] takes index as value, looks at out_grad[a, b] takes vector
        # and adds the vector at dembeddings[index]
        return self.dembeddings
    
    def step(self, learning_rate, clip_val=1.0):
        cp.clip(self.dembeddings, -clip_val, clip_val, self.dembeddings)
        
        self.embeddings -= learning_rate * self.dembeddings
        
        # Idk if its possible to get dinputs bc they are indices
        

class RNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        # Set RNN params
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Storage
        self.input_cache: Tensor | None = None
        
        # Initialize weights
        self.W_xh: Tensor = cp.random.randn(input_dim, hidden_dim) * cp.sqrt(1 / input_dim)
        self.W_hh: Tensor = cp.random.randn(hidden_dim, hidden_dim) * cp.sqrt(1 / hidden_dim)
        self.W_hy: Tensor = cp.random.randn(hidden_dim, output_dim) * cp.sqrt(1 / hidden_dim)
        
        # Initialize bias to 0
        self.bh: Tensor = cp.zeros((hidden_dim))
        self.by: Tensor = cp.zeros((output_dim))
        
    def params(self):
        return [
            (self.W_xh, self.dW_xh),
            (self.W_hh, self.dW_hh),
            (self.W_hy, self.dW_hy),
            (self.bh,   self.dbh),
            (self.by,   self.dby)
        ]
        
        
    # Key operation for RNN is: h_t = act(h_t=1 @ W_hh + x_t @ W_xh + b)
    
    def forward(self, x, h_prev=None) -> Tensor:
        self.input_cache = x
        B, seq_len = x.shape[:2]
        self.seq_len = seq_len
        self.batch_size = B
        
        if h_prev is None:
            h_prev: Tensor = cp.zeros((B, self.hidden_dim))
            
        self.init_h_prev = h_prev
            
        self.hidden_states = []
        
        for t in range(seq_len):
            x_t = x[:, t, :] # (B, input_dim)
            
            prev_part = h_prev @ self.W_hh # (B, hidden_dim) @ (hidden_dim, hidden_dim) -> (B, hidden_dim) 
            next_part = x_t @ self.W_xh # (B, input_dim) @ (input_dim, hidden_dim) -> (B, hidden_dim)
            
            combined_part = prev_part + next_part + self.bh
            h_next = cp.tanh(combined_part) # (B, hidden_dim) + (B, hidden_dim) + (hidden_dim)
            
            self.hidden_states.append(h_next)
            h_prev = h_next
            
        self.hidden_states = cp.stack(self.hidden_states, axis=1) # (B, seq_len, hidden_dim)
        output = self.hidden_states @ self.W_hy + self.by # (B, seq_len, hidden_dim) @ (hidden_dim, out_dim) + (out_dim) -> (B, seq_len, out_dim)
        
        return output, h_next
            
            
    
    def backward(self, dlogits: Tensor):
        # dlogits is (B, seq_len, out_dim)
        
        # Accumulation part (zero out at the start of each) so we dont accumulate
        self.dW_xh = cp.zeros_like(self.W_xh)
        self.dW_hh = cp.zeros_like(self.W_hh)
        self.dbh = cp.zeros_like(self.bh)
        
        # Possible to calculate off the cuff
        self.dhidden_states = dlogits @ self.W_hy.T
        
        # The part below works because we have to sum up the influences of W_hy across logits in all batches,
        # and flattened dot product essentially adds also cross-batch
        flat_hidden = self.hidden_states.reshape(-1, self.hidden_dim) # (B * seq_len, hidden_dim)
        flat_dlogits = dlogits.reshape(-1, self.output_dim) # (B * seq_len, out_dim)
        self.dW_hy = flat_hidden.T @ flat_dlogits  # (hidden_dim, B * seq_len) @ (B * seq_len, out_dim)
        self.dby = cp.sum(dlogits, axis=(0, 1)) # (out_dim)
        
        # Init the next step to run backwards our forward loop
        dh_next = cp.zeros((self.batch_size, self.hidden_dim))
        
        # Array for holding change wrt. embeds (x_ts)
        dx_ts = []
        
        for t in range(self.seq_len - 1, -1, -1):
            x_t = self.input_cache[:, t, :] # (B, input_dim)
            h_t = self.hidden_states[:, t, :] # (B, hidden_dim)
            dh_out_t = self.dhidden_states[:, t, :] # (B, hidden_dim) # Only the gradient from current output
            dh_total_t = dh_out_t + dh_next # (B, hidden_dim) combines impact on output and future outputs. Zero impact at t=last, max at t=0
            h_prev_t = self.hidden_states[:, t - 1, :] if t != 0 else self.init_h_prev
            # dtanh/dx = 1 - tanh^2(x)
            dh_preact_t = (1 - h_t ** 2) * dh_total_t # How current preact h (composed of the double weight matmul) impacts current t and t + 1
            # we need dL/dpreact so its dact/dpreact * dL/dact
            # The activated part simply goes into hidden states. Its part of our output. So we multiply by grad of this part
            # This is our prev_part + next_part + self.bh. Shape (B, hidden_dim)
            # Prev part and next part are same shapes. The grad for addition is 1 * out_grad (dh_preact_t) = out_grad
            dprev_part_t = dh_preact_t 
            dnext_part_t = dh_preact_t
            self.dbh += cp.sum(dh_preact_t, axis=0) # (hidden_dim)
            
            # impacts current output and future output.
            dh_prev = dprev_part_t @ self.W_hh.T # how t - 1 influences t and ones after till the end
            # we then set dh_total t in next iteration to how it influences contemporary state and how it influences t+1 and so on
            # we add in step t - 1 the effect of h on t - 1 and on t+1-1 and so on, totaling whole network.
            
            self.dW_hh += h_prev_t.T @ dprev_part_t
            self.dW_xh += x_t.T @ dnext_part_t
            
            # Optionally if we will have embeds:
            dx_t = dnext_part_t @ self.W_xh.T # (B, input_dim)
            dx_ts.append(dx_t)
            
            dh_next = dh_prev
            # The dh_next is how t influences  t+1 and further. dh_total consists of current impact of h and the future ones. 
            # dh_prev consists of how t - 1 influences dh_total, so how t - 1 impacts t, t+1. 
            # In next iteration when t = t - 1 our h_prev is actually the future and we add the contemporary impact (t) + h_next 
            # which were current + future one step ago, but now its just future.
            
            # if the gradient explodes DURING backwards pass then we might have to clip it inside BPTT
            #np.clip(dh_prev, -1, 1, out=dh_prev)
            
        dx_ts = cp.stack(dx_ts, axis=1) # (B, seq_len, input_dim)
        
        return dx_ts
        
        
    def step(self, learning_rate, clip_val=1.0):
        # Clip grads before substraction
        for p, grad in self.params():
            cp.clip(grad, -clip_val, clip_val, out=grad)
            # the out means we do it in place (or points just to the location)
            
        # Update parameters
        self.W_xh -= learning_rate * self.dW_xh
        self.W_hh -= learning_rate * self.dW_hh
        self.bh -= learning_rate * self.dbh
        
        self.W_hy -= learning_rate * self.dW_hy
        self.by -= learning_rate * self.dby
        
        
        
        