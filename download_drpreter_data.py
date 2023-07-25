import argparse
from data_utils import Downloader


# python download_drpreter_data.py --data_path comp/DRPreter/Data/
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='ProgramName', description='What the program does')
    parser.add_argument('--metric',  default='ic50', help='')
    parser.add_argument('--run_id',  default='0', help='')
    parser.add_argument('--epochs',  default=1, help='')
    # parser.add_argument('--data_split_seed',  default=1, help='')
    parser.add_argument('--data_split_id',  default=0, help='')
    # parser.add_argument('--encoder_type',  default='gnn', help='')
    parser.add_argument('--data_path',  default='', help='')
    parser.add_argument('--data_type',  default='CCLE', help='')
    parser.add_argument('--data_version',  default='benchmark-data-imp-2023', help='')
    # parser.add_argument('--out_dir',  default='', help='')
    # parser.add_argument('--feature_path',  default=None, help='')

    args = parser.parse_args()

    dw = Downloader(args.data_version)
    dw.download_candle_data(data_type=args.data_type, split_id=args.data_split_id, data_dest=args.data_path)
