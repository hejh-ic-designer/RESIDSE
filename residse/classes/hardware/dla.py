from residse.classes.hardware.memory import Memory

class Dla:
    def __init__(self, mac_unroll: dict, a_buf: dict, w_buf: dict, dram: dict):
        self.mac_unroll = mac_unroll
        self.a_buf = Memory(a_buf, True)
        self.w_buf = Memory(w_buf, False)
        self.dram = Memory(dram, False)
        self.parse_mac()


    def parse_mac(self):        
        self.number_of_unroll = len(self.mac_unroll)
        self.unroll_dim = []
        self.unroll_dim_size = []
        self.number_of_mac = 1
        for dim, dim_size in self.mac_unroll.items():
            self.unroll_dim.append(dim)
            self.unroll_dim_size.append(dim_size)
            self.number_of_mac *= dim_size

        # unroll
        self.u_h  = self.mac_unroll.get('h' , 1)
        self.u_w  = self.mac_unroll.get('w' , 1)
        self.u_oc = self.mac_unroll.get('oc', 1)
        self.u_ic = self.mac_unroll.get('ic', 1)
        self.u_fx = self.mac_unroll.get('fx', 1)
        self.u_fy = self.mac_unroll.get('fy', 1)

