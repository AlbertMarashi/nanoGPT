batch_size = 4
block_size = 16
loss at iter 1000 = 2.988


====
batch_size = 12
block_size = 64
n_embd = 128
loss at iter 500 = 2.3347
loss at iter 1000 = 2.0808
loss at iter 1500 = 1.9451
loss at iter 2000 = 1.8208
max_thinking_steps = 1
====
batch_size = 12
block_size = 64
n_embd = 128
loss at iter 500 = 2.3240
loss at iter 1000 = 2.0982
loss at iter 1500 = 1.9980
loss at iter 2000 = 2.0122
max_thinking_steps = 5
====
batch_size = 12
block_size = 32
loss at iter =
max_thinking_steps = 10



==========
==========
n_layer = 3
n_head = 4
n_embd = 128
batch_size = 12
block_size = 64
n_embd = 128
target_usage = 0.5
====
max_thinking_steps = 5
step 500: train loss 2.3492, val loss 2.3520
step 1000: train loss 2.1049, val loss 2.1489
step 1500: train loss 1.9960, val loss 2.0485
step 2000: train loss 1.9353, val loss 2.0116
====
max_thinking_steps = 1
step 500: train loss 2.3297, val loss 2.3339
step 1000: train loss 2.0901, val loss 2.1389
step 1500: train loss 1.9889, val loss 2.0429
step 2000: train loss 1.9278, val loss 2.0073
----
max_thinking_steps = 5
based on previous confidence
step 500: train loss 2.3662, val loss 2.3648
step 1000: train loss 2.1122, val loss 2.1461
step 1500: train loss 2.0111, val loss 2.0600
step 2000: train loss 1.9495, val loss 2.0259


==========
Using new entropy loss functionality
max_thinking_steps = 5
step 500: train loss 2.3790, val loss 2.3770
step 1000: train loss 2.1315, val loss 2.1645