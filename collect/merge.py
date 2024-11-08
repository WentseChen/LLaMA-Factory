import json

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--file_path', type=str, default="/zfsauton2/home/wentsec/incontext_RL/test", help='where to save the data')
parser.add_argument('--batch_size', type=int, default=256, help='number batch size for each iteration')
parser.add_argument('--phase', type=int, default=1, help='phase number')
args = parser.parse_args()

all_data = []
for i in range(8):
    file_name = args.file_path + "/phase{}_rank{}.json".format(args.phase, i)
    with open(file_name, "r") as f:
        data = json.load(f)
    all_data.append(data)

output_data = dict()
for data in all_data:
    for key in data:
        if key not in output_data:
            output_data[key] = data[key]
        else:
            output_data[key] += data[key]
    
file_name = args.file_path + "/phase{}.json".format(args.phase)
with open(file_name, "w") as f:
    json.dump(output_data, f)