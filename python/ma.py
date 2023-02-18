device = None

def print_device():
    print(device)

def main(device):
    # 这是函数内的不可见变量
    device = device
    print_device()

if __name__ == '__main__':
    # 这不是函数, 所以是函数外面的可见变量
    device = 10
    print_device()
    