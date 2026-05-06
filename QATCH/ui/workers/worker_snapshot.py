class WorkerSnapshot:
    def __init__(self, t, d1, d2):
        self.t = t
        self.d1 = d1
        self.d2 = d2

    def get_t1_buffer(self, idx):
        return self.t

    def get_d1_buffer(self, idx):
        return self.d1

    def get_d2_buffer(self, idx):
        return self.d2
