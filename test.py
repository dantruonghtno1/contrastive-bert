from config import Param

def main():
    param = Param()
    args = param.args
    print(args.data_path)


if __name__ == "__main__":
    main()
