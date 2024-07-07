import time
import json
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np

from utils import AverageMeter, calculate_accuracy, calculate_precision_and_recall, calculate_mse_and_mae
from scipy import stats


def get_video_results(outputs, class_names, output_topk):
    sorted_scores, locs = torch.topk(outputs,
                                     k=min(output_topk, len(class_names)))

    video_results = []
    for i in range(sorted_scores.size(0)):
        video_results.append({
            'label': class_names[locs[i].item()],
            'score': sorted_scores[i].item()
        })

    return video_results


def inference(data_loader, model, logger, inf_json, device):
    print('inference')

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()


    end_time = time.time()
    accuracies = AverageMeter()
    results = {"result": defaultdict(list)}

    mae, mse, num_examples = 0., 0., 0
    s_predict = []
    s_target = []
    with torch.no_grad():
        for i, (inputs, image_name, targets) in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            # video_ids, segments = zip(*targets)
            
            # target_label = target_label.to(device, non_blocking=True)
            outputs, _ = model(inputs)
            targets = targets.to(device, non_blocking=True)
            
            num_examples += targets.size(0)
            acc = calculate_accuracy(outputs, targets)
            # precision, recall = calculate_precision_and_recall(outputs, targets)
            mse_o, mae_o = calculate_mse_and_mae(outputs, targets)
            mse += mse_o
            mae += mae_o
            # mae = mae / num_examples
            # mse = mse / num_examples
  

            accuracies.update(acc, inputs.size(0))


            outputs = F.softmax(outputs, dim=1)
            # outputs = outputs.sigmoid().cpu()
            outputs = outputs.cpu()
            outputs_value = np.argmax(outputs)


            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('[{}/{}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                      i + 1,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                        acc=accuracies))
            targets_r = targets.cpu().numpy()[0].astype(np.float64)
            outputs_value_r = outputs_value.numpy().astype(np.float64)
            outputs_value_r = outputs_value_r.tolist()
            outputs_r = outputs.cpu().numpy()[0]
            outputs_r = list(outputs_r)
            outputs_r = [float(i) for i in outputs_r]
            acc_r = acc
            # print(type(targets_r),type(outputs_value_r), type(outputs_r), type(outputs_r[0]), type(acc_r))
            results['result'][image_name[0]].append({'target': targets_r, 'output_value': outputs_value_r, 'output': outputs_r, 'acc': acc_r, 'mse': mse, "mae": mae})
            # results['result']['image_name'][i] = image_name[0]
            # results['result']['target'][i] = targets.item()
            # results['result']['output_value'][i] = outputs_value.item()
            # results['result']['output'][i] = outputs.item()
            # results['result']['acc'][i] = acc
            s_predict.append(outputs_value.cpu().numpy())
            s_target.append(targets.cpu().numpy())
            logger.log({'image_name': image_name, 'target': targets.cpu().numpy(), 'output_value': outputs_value.cpu().numpy(), 'output': outputs.cpu().numpy(), 'acc': acc, 'mse': mse, "mae": mae})

    s_predict = np.array(s_predict)
    s_target = np.array(s_target)
    res = stats.spearmanr(s_predict, s_target)
    print(res.statistic)
    with inf_json.open('w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


