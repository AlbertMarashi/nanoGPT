# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-shakespeare-char'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 6
block_size = 64 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 2
n_head = 4
n_embd = 64
dropout = 0.1

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 2000
lr_decay_iters = 2000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 0 # not super necessary potentially

max_thinking_steps = 1  # Add this
thinking_budget_decay_rate = 0.99
budget_factor = 0.01  # Add this
target_usage = 0.5  # Add this

# on macbook also add
device = 'cpu'  # run on cpu only
compile = False # do not torch compile the model
