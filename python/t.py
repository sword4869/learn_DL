import hello
# python是逐行执行的，所以当它读到import hello的时候，也会执行hello.py
# 所以运行了hello中第一行的print
 
if __name__ == "__main__":
    print ('This is main of module "world.py"')
    hello.sayHello()
    print(__name__)
    