from configs import BaseConfig


def main():
    configs = BaseConfig.from_args()
    print(f'Batch size: {configs.train.batch_size}')
    print(f'Hidden size: {configs.model.hidden_size}')


if __name__ == '__main__':
    main()
