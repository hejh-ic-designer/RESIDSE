import json
from residse.classes.hardware.dla import Dla



class HardwareGenerator:
    '''use get_dla method to return a dla'''
    def __init__(self, json_hw: str):
        ''' json_hw: path to json format dla path'''
        self.parse_json_hw(json_hw)
        

    def get_dla(self):
        return self.dla

    def parse_json_hw(self, json_hw: str):
        assert isinstance(json_hw, str), 'json_hw is the path to your json format dla input.'
        with open(json_hw) as f:
            json_di = json.load(f)
        self.json_di = json_di
        
        # start extract hw info
        mac_unroll: dict = self.json_di["mac_unroll"]
        a_buf: dict = self.json_di["a_buf"]
        w_buf: dict = self.json_di["w_buf"]
        dram: dict = self.json_di["dram"]
        self.dla = Dla(mac_unroll, a_buf, w_buf, dram)


if __name__ == '__main__':
    json_hw = 'residse/inputs/HW/res18_example1.json'
    example_1_dla = HardwareGenerator(json_hw).get_dla()
    print(example_1_dla.number_of_mac)
    print(example_1_dla.number_of_unroll)
    print(example_1_dla.unroll_dim)
    print(example_1_dla.unroll_dim_size)
    print(example_1_dla.mac_unroll)
    print(example_1_dla.a_buf.size_list)
    print(example_1_dla.w_buf.size)
    print(example_1_dla.dram.bw)
