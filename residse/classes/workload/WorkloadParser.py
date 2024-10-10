import yaml
from residse.classes.workload.stack import Stack

class WorkloadParser:
    def __init__(self, yaml_path):
        stack_info_lst = self.parse_yaml(yaml_path)
        self.stacks = [Stack(id, stack_di) for id, stack_di in enumerate(stack_info_lst)]
    
    def parse_yaml(self, yaml_path):
        with open(yaml_path, 'r') as f:
            docs = list(yaml.safe_load_all(f))
        return docs
    
    def get_stacks(self):
        return self.stacks


if __name__ == '__main__':
    # yaml_path = 'residse/inputs/WL/resnet18.yml'
    yaml_path = 'residse/inputs/WL/sesr.yml'
    # yaml_path = 'residse/inputs/WL/edgeSRGAN.yml'
    stacks = WorkloadParser(yaml_path).get_stacks()
    for stack in stacks:
        print(stack)