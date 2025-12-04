import argparse
from dataloader.XJTU_loader import XJTUDdataset
from dataloader.MIT_loader import MITDdataset
from nets.Model import SOHMode
import os
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_args():
    parser = argparse.ArgumentParser(description='A benchmark for SOH estimation')
    parser.add_argument('--random_seed', type=int, default=2023)
    # data
    parser.add_argument('--data', type=str, default='XJTU',choices=['XJTU','MIT'])
    parser.add_argument('--input_type',type=str,default='charge',choices=['charge','partial_charge','handcraft_features'])
    parser.add_argument('--test_battery_id',type=int,default=1,help='test battery id, 1-8 for XJTU (1-15 for batch-2), 1-5 for MIT')
    parser.add_argument('--batch_size',type=int,default=128)
    parser.add_argument('--normalized_type',type=str,default='minmax',choices=['minmax','standard'])
    parser.add_argument('--minmax_range',type=tuple,default=(-1,1),choices=[(0,1),(-1,1)])
    parser.add_argument('--batch', type=int, default=1,choices=[1,2,3,4,5,6,7,8,9])

    # model
    parser.add_argument('--model',type=str,default='CNN',choices=['CNN','LSTM','GRU','MLP','Attention'])
    # CNN lr=2e-3  early_stop=30

    parser.add_argument('--lr',type=float,default=2e-3)
    parser.add_argument('--weight_decay', default=5e-4)
    parser.add_argument('--n_epoch',type=int,default=100)
    parser.add_argument('--early_stop',type=int,default=30)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--save_folder',default='results')
    parser.add_argument('--experiment_num',default=1,type=int,help='The number of times you want to repeat the same experiment')

    args = parser.parse_args()
    return args

def load_data(args,test_battery_id):
    if args.data == 'XJTU':
        loader = XJTUDdataset(args)
    else:
        loader = MITDdataset(args)

    if args.input_type == 'charge':
        data_loader = loader.get_charge_data(test_battery_id=test_battery_id)
    elif args.input_type == 'partial_charge':
        data_loader = loader.get_partial_data(test_battery_id=test_battery_id)
    else:
        data_loader = loader.get_features(test_battery_id=test_battery_id)
    return data_loader

def main(args):
    for batch in [1]:
        ids_list = [1,2,3,4,5,6,7,8]
        for test_id in ids_list:
            setattr(args,'batch',batch)
            for e in range(3):
                print()
                print(args.normalized_type,args.minmax_range,args.model,args.data, args.input_type,batch,test_id,e)
                try:
                    data_loader = load_data(args, test_battery_id=test_id)
                    model = SOHMode(args)
                    model.Train(data_loader['train'], data_loader['valid'], data_loader['test'],
                                save_folder=f'results/{args.data}-{args.input_type}/{args.model}/batch{batch}-testbattery{test_id}/experiment{e+1}',
                                )
                    del model
                    del data_loader
                    torch.cuda.empty_cache()

                except:
                    torch.cuda.empty_cache()
                    continue

def multi_task_XJTU():  # 一次性训练所有模型和所有输入类型
    args = get_args()
    setattr(args,'data','XJTU')
    # 'CNN','MLP', ,'LSTM','GRU'   'handcraft_features','charge',
    for m in ['Attention']:
        for type in ['partial_charge']:
            setattr(args, 'model', m)
            setattr(args, 'input_type',type)
            main(args)
            torch.cuda.empty_cache()

# 照table10来跑-XJTU-charge-[-1,1]
def multi_task_XJTU_for_table_C10():
    args = get_args()
    setattr(args, 'data', 'XJTU')
    setattr(args, 'normalized_type', 'minmax')   # min-max
    setattr(args, 'input_type','charge')         # charge
    setattr(args, 'minmax_range', (-1, 1))       # [-1,1]
    setattr(args, 'input_type', 'charge')        # complete charging data
    setattr(args, 'n_epoch', 100)
    setattr(args, 'early_stop', 30)
    setattr(args, 'lr',2e-3)

    # 5 个模型, 'LSTM', 'GRU', 'MLP', 'Attention'
    for m in ['CNN']:
        setattr(args, 'model', m)
        main_for_all_batches_xjtu(args)   
        torch.cuda.empty_cache()

def main_for_all_batches_xjtu(args):
    # XJTU 各个 batch 的电池数
    batch_to_ids = {
        1: range(1, 9),
        2: range(1, 16),
        3: range(1, 9),
        4: range(1, 9),
        5: range(1, 9),
        6: range(1, 9),
        # 7: range(1, 9),
    }

    for batch, ids_list in batch_to_ids.items():
        setattr(args, 'batch', batch)
        for test_id in ids_list:
            for e in range(3):  # 3 次实验，对应表里的平均
                print()
                print(args.normalized_type, args.minmax_range,
                      args.model, args.data, args.input_type,
                      batch, test_id, e)
                try:
                    data_loader = load_data(args, test_battery_id=test_id)
                    model = SOHMode(args)
                    model.Train(
                        data_loader['train'],
                        data_loader['valid'],
                        data_loader['test'],
                        save_folder=(
                            f'results/{args.data}-{args.input_type}/'
                            f'{args.model}/batch{batch}-testbattery{test_id}/experiment{e+1}'
                        ),
                    )
                    del model
                    del data_loader
                    torch.cuda.empty_cache()
                except:
                    torch.cuda.empty_cache()
                    continue


def multi_task_XJTU_table_C10(args):
    """
    按 Table C.10 的设定跑：
    - data 固定 XJTU
    - 归一化方式固定 minmax
    - minmax_range 固定 (-1, 1)

    python main.py \
    --model CNN \
    --lr 2e-3 \
    --input_type charge \
    --batch 2 \
    --test_battery_id 1 \
    --experiment_num 3

    全部由命令行指定。
    """
    # 固定与 Table C.10 对齐的部分
    setattr(args, 'data', 'XJTU')
    setattr(args, 'normalized_type', 'minmax')   # min-max
    setattr(args, 'minmax_range', (-1, 1))       # [-1,1]

    # ---- 单电池、可重复多次实验 ----
    # 使用命令行提供的 batch / test_battery_id / experiment_num
    batch = args.batch
    test_id = args.test_battery_id

    print(f"[Table C10模式] data={args.data}, input_type={args.input_type}, "
          f"model={args.model}, lr={args.lr}, "
          f"norm={args.normalized_type}, range={args.minmax_range}, "
          f"batch={batch}, test_battery_id={test_id}, "
          f"n_epoch={args.n_epoch}, early_stop={args.early_stop}, "
          f"experiment_num={args.experiment_num}")

    setattr(args, 'batch', batch)

    for e in range(args.experiment_num):
        print()
        print(args.normalized_type, args.minmax_range,
              args.model, args.data, args.input_type,
              batch, test_id, e)

        try:
            data_loader = load_data(args, test_battery_id=test_id)
            model = SOHMode(args)
            model.Train(
                data_loader['train'],
                data_loader['valid'],
                data_loader['test'],
                save_folder=(
                    f"results/{args.data}-{args.input_type}/"
                    f"{args.model}/batch{batch}-testbattery{test_id}/experiment{e + 1}"
                ),
            )
            del model
            del data_loader
            torch.cuda.empty_cache()
        except Exception as ex:
            print("!! experiment failed with exception:", ex)
            torch.cuda.empty_cache()
            continue


def run_xjtu_tableC10_from_batch(args):
    """
    功能：
    - 按 Table C.10 的设定（XJTU + minmax + [-1,1]）跑
    - 从 args.batch 开始，一直跑到最后一个 batch（目前是 6）
    - 每个 batch 里把所有 test_battery_id 都跑完
    - 每个 battery 重复 args.experiment_num 次实验

    用法示例：
    python main.py \
        --model Attention \
        --lr 2e-3 \
        --input_type partial_charge \
        --batch 3 \
        --experiment_num 3
    表示：从 batch2 开始，batch2~batch6 都跑完，每个电池跑 3 次实验。
    """
    # 固定为 XJTU + minmax + [-1,1]，和 multi_task_XJTU_table_C10 一致
    setattr(args, 'data', 'XJTU')
    setattr(args, 'normalized_type', 'minmax')
    setattr(args, 'minmax_range', (-1, 1))

    # XJTU 各 batch 的电池编号范围
    batch_to_ids = {
        1: range(1, 9),   # batch1: 1-8
        2: range(1, 16),  # batch2: 1-15
        3: range(1, 9),
        4: range(1, 9),
        5: range(1, 9),
        6: range(1, 9),

    }

    # 起始 batch：沿用命令行里的 --batch 参数
    batch_start = args.batch
    if batch_start not in batch_to_ids:
        raise ValueError(f"batch_start={batch_start} 不在可用范围 {list(batch_to_ids.keys())} 内")

    print(f"[批量 TableC10 模式] 从 batch{batch_start} 开始")
    print(f"  data={args.data}, input_type={args.input_type}, "
          f"model={args.model}, lr={args.lr}, "
          f"norm={args.normalized_type}, range={args.minmax_range}, "
          f"n_epoch={args.n_epoch}, early_stop={args.early_stop}, "
          f"experiment_num={args.experiment_num}")

    # 从 batch_start 一直跑到最后一个 batch
    for batch in sorted(batch_to_ids.keys()):
        if batch < batch_start:
            continue

        ids_list = batch_to_ids[batch]
        setattr(args, 'batch', batch)

        print(f"\n==== 开始 batch {batch}，电池列表: {list(ids_list)} ====")

        for test_id in ids_list:
            for e in range(args.experiment_num):
                print()
                print("[RUN]", args.normalized_type, args.minmax_range,
                      args.model, args.data, args.input_type,
                      f"batch={batch}", f"test_id={test_id}", f"exp={e}")

                try:
                    data_loader = load_data(args, test_battery_id=test_id)
                    model = SOHMode(args)
                    model.Train(
                        data_loader['train'],
                        data_loader['valid'],
                        data_loader['test'],
                        save_folder=(
                            f"results/{args.data}-{args.input_type}/"
                            f"{args.model}/batch{batch}-testbattery{test_id}/experiment{e + 1}"
                        ),
                    )
                    del model
                    del data_loader
                    torch.cuda.empty_cache()
                except Exception as ex:
                    print(f"!! [WARNING] batch{batch} battery{test_id} exp{e+1} 失败，异常: {ex}")
                    torch.cuda.empty_cache()
                    continue


def multi_task_MIT():

    args = get_args()
    for norm in ['standard','minmax']:  # normalized_type
        setattr(args,'normalized_type',norm)
        setattr(args,'minmax_range',(0,1))
        setattr(args,'data','MIT')
        for m in ['GRU']:  # model
            for type in ['partial_charge']:
                setattr(args, 'model', m)
                setattr(args, 'input_type', type)
                for batch in range(1,10):   # batch
                    ids_list = [1, 2, 3, 4, 5]
                    for test_id in ids_list:   # test_battery_id
                        setattr(args, 'batch', batch)
                        for e in range(5):   # experiment
                            print()
                            print(args.model, args.data, args.input_type, batch, test_id, e)
                            try:
                                data_loader = load_data(args, test_battery_id=test_id)
                                model = SOHMode(args)
                                model.Train(data_loader['train'], data_loader['valid'], data_loader['test'],
                                            save_folder=f'results/{norm}/{args.data}-{args.input_type}/{args.model}/batch{batch}-testbattery{test_id}/experiment{e + 1}',
                                            )
                                del model
                                del data_loader
                                torch.cuda.empty_cache()
                            except:
                                torch.cuda.empty_cache()
                                continue

                torch.cuda.empty_cache()


if __name__ == '__main__':
    # if just want to train one model on one battery and one input type, use this:
    # args = get_args()
    # for e in range(args.experiment_num):
    #     data_loader = load_data(args, test_battery_id=args.test_battery_id)
    #     model = SOHMode(args)
    #     model.Train(data_loader['train'], data_loader['valid'], data_loader['test'],
    #                 save_folder=f'results/{args.data}-{args.input_type}/{args.model}/batch{args.batch}-testbattery{args.test_battery_id}/experiment{e + 1}',
    #                 )

    # multi_task_XJTU_for_table_C10()
    
    # 从命令行读取所有参数
    args = get_args()
    # multi_task_XJTU_table_C10(args)

    run_xjtu_tableC10_from_batch(args)

    # multi_task_XJTU()
    # multi_task_MIT()

# 在程序中，我们提供了100个电池（我们自己实验生成了55个+MIT的45个）, with 3 input types and 3 normalization methods, 5 种基本模型供你选择
# 你可以自己选择训练哪些模型，哪些输入类型，哪些电池，哪些归一化方法，哪些超参数，哪些训练策略
# 你可以自己选择训练多少次，每次训练的结果都会保存在results文件夹中
# 我们这里只是提供一个baseline，你可以在此基础上进行改进，比如使用更好的模型，更好的超参数，更好的训练策略等等
