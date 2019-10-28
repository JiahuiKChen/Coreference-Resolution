import re
import os
import copy
import argparse
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('key_dir', type=str, help='key file directory')
arg_parser.add_argument('response_dir', type=str, help='response file directory')
arg_parser.add_argument('response_file', type=str, help='response file IDs')
arg_parser.add_argument('-v', '--verbose', action='store_true', required=False, help='if true print correct mentions as well')
args = arg_parser.parse_args()


def intersection(key_cluster, response_cluster):
    tmp = copy.deepcopy(key_cluster)
    matched_response, unmatched_response = [], []
    for response_mention in response_cluster:
        sent_id, span = response_mention
        for i, key_mention in enumerate(tmp):
            if key_mention:
                key_sent_id, max_span, min_span = key_mention
                if sent_id == key_sent_id and \
                   min_span in span and span in max_span:
                    tmp[i] = None
                    matched_response.append(response_mention)
                    break
        else:
            unmatched_response.append(response_mention)
    missed_key = [x for x in tmp if x]
    return matched_response, unmatched_response, missed_key


def main():
    key_dir = args.key_dir.strip('/')
    response_dir = args.response_dir.strip('/')
    # file_ids = set([x.rstrip('.key') for x in os.listdir(key_dir) if x.endswith('.key')])
    file_ids = [x.strip() for x in open(args.response_file, 'r').read().strip().splitlines() if x.strip()]
    response_cluster_len = []
    key_cluster_len = []
    inter_count = []
    for file_id in sorted(file_ids):
        key_path = '{}/{}.key'.format(key_dir, file_id)
        response_path = '{}/{}.response'.format(response_dir, file_id)
        print('\n'+'#' * (len(file_id)+10))
        print('#    {}    #'.format(file_id))
        print('#' * (len(file_id)+10)+'\n')
        if not os.path.exists(response_path):
            print('{} not exists!'.format(response_path))
            continue
        # Read response/key files
        with open(response_path, 'r') as f:
            resp_cluster_list = f.read().strip().split('\n\n')
        with open(key_path, 'r') as f:
            key_cluster_list = f.read().strip().split('\n\n')
        # Eval on each key cluster 
        for cluster in key_cluster_list:
            first_appearance = cluster.strip().splitlines()[0].strip()
            key_lines = [x.strip() for x in cluster.strip().splitlines()[1:]]
            response_lines = []
            for c2 in resp_cluster_list:
                if not c2.strip():
                    continue
                fa2 = c2.strip().splitlines()[0].strip()
                if first_appearance == fa2:
                    response_lines = [x.strip() for x in c2.strip().splitlines()[1:]]
                    break
            key_cluster = []
            response_cluster = []
            for key_line in key_lines:
                sent_id, max_span, min_span = re.findall(r'{(.+?)}', key_line, re.DOTALL)
                key_cluster.append((sent_id, max_span, min_span))
            for response_line in response_lines:
                sent_id, span = re.findall(r'{(.+?)}', response_line, re.DOTALL)
                response_cluster.append((sent_id, span))
            # Map responses to keys
            matched_response, unmatched_response, missed_key = intersection(key_cluster, response_cluster)
            inter_count.append(len(matched_response))
            response_cluster_len.append(len(response_cluster))
            key_cluster_len.append(len(key_cluster))
            # Print errors to the screen
            if inter_count[-1] != response_cluster_len[-1] or inter_count[-1] != key_cluster_len[-1] or args.verbose:
                print('-' * 50)
                print('| '+first_appearance)
                if args.verbose:
                    for mention in matched_response:
                        print('| Correct Mention: '+' '.join(['{{{}}}'.format(x) for x in mention]))
                for mention in unmatched_response:
                    print('| Wrong Mention: '+' '.join(['{{{}}}'.format(x) for x in mention]))
                for mention in missed_key:
                    print('| Missed Mention: '+' '.join(['{{{}}}'.format(x) for x in mention]))
                print('-' * 50 + '\n')
    # micro average precision/recall
    mp = 1.0 * sum(inter_count) / sum(response_cluster_len) if sum(response_cluster_len)>0 else 0
    mr = 1.0 * sum(inter_count) / sum(key_cluster_len)
    print('Micro Average Precision: {:.5f} ({:d}/{:d})'.format(mp, sum(inter_count), sum(response_cluster_len)))
    print('Micro Average Recall:    {:.5f} ({:d}/{:d})'.format(mr, sum(inter_count), sum(key_cluster_len)))
    print('F-score:                 {:.5f}'.format(2*mp*mr/(mp+mr) if abs(mp+mr)>1e-5 else 0))


if __name__ == '__main__':
    main()
