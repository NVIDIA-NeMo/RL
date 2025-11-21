from ._imports import ParallelDecoder, get_num_transfer_tokens, get_transfer_index, MASK_ID, EOS_ID

class FixedParallelDecoder(ParallelDecoder):
    """ 
    This decoder decodes tokens in a fixed number of steps.
    Adapted from _FixedParallelDecoder in soft_token_experiment.py.
    """
    def __init__(self, temperature, steps, remasking='low_confidence', mask_id=MASK_ID, eos_id=EOS_ID):
        super().__init__(temperature, remasking, mask_id)
        self.steps = steps
        self.iter = 0
        self.eos_id = eos_id

    def block_init(self, block_x, block_id):
        # TODO(zhengda) we need to handle steps correctly here when the distributed version changes the gen length.
        block_mask_index = block_x == self.mask_id
        self.num_transfer_tokens = get_num_transfer_tokens(block_mask_index, self.steps)
        self.iter = 0

    def decode(self, logits, block_start, block_end, x, iter_threshold = None):
        """ Decode the logits in a block.
        """
        mask_index = (x[block_start:block_end] == self.mask_id)
        assert mask_index.shape[1] == logits.shape[1]

        curr_x = x[block_start:block_end]
        x0, transfer_index = get_transfer_index(logits, self.temperature, self.remasking, mask_index, curr_x, self.num_transfer_tokens[:, self.iter], None)
        self.iter += 1
        x[block_start:block_end][transfer_index] = x0[transfer_index]