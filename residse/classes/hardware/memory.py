class Memory:
    def __init__(self, mem_dict: dict, is_iterate: bool):
        self.size = mem_dict.get("size", None)
        self.bw = mem_dict.get("bandwidth", None)
        if is_iterate:
            size_lower_limit = mem_dict.get("lower_limit", None)   # MB
            size_upper_limit = mem_dict.get("upper_limit", None)   # MB
            size_step = mem_dict.get("size_step", None)
            size_points = mem_dict.get("size_points", None)
            self.calc_size_list(size_lower_limit, size_upper_limit, size_step, size_points)


    def calc_size_list(self, size_lower_limit, size_upper_limit, size_step, size_points):
        """
        three cases:
        1. lower_limit, size_step, size_points --> size_list
        2. lower_limit, upper_limit, size_points --> size_list
        3. lower_limit, uppper_limit, size_step --> size_list
        """
        if (size_lower_limit and size_step and size_points):
            pass
        elif (size_lower_limit and size_upper_limit and size_points):
            size_step = (size_upper_limit - size_lower_limit) / (size_points - 1)
        elif (size_lower_limit and size_upper_limit and size_step):
            size_points = int((size_upper_limit - size_lower_limit) / size_step) + 1
        else :
            raise SyntaxError("memory items set ERROR!")

        self.size_list = [size_lower_limit + i * size_step for i in range(size_points)]

    def get_size_list(self):
        return self.size_list
    
    
if __name__ == '__main__':

    a_buf_di = {
        "lower_limit": 0.5,
        "size_step": 0.5,
        "size_points": 10}
    abuf = Memory(a_buf_di, True)
    print(abuf.get_size_list())


