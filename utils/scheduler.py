
class Scheduler(object):
    def __init__(self, start_value, end_value, start_iter, end_iter, mode=None):
        self.start_value = start_value
        self.end_value = end_value
        self.start_iter = start_iter
        self.end_iter = end_iter
        self.mode = mode

    def get_value(self, current_iter):
        if self.mode is None:
            return self.end_value
        elif self.mode == 'linear':
            current_iter = min(max(current_iter, self.start_iter), self.end_iter)
            return self.start_value + float(current_iter-self.start_iter)/float(self.end_iter-self.start_iter) \
                   * (self.end_value - self.start_value)
        else:
            raise NotImplementedError('mode %s is not implemented' % self.mode)