import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import pandas as pd

from sklearn.metrics import roc_auc_score, classification_report,average_precision_score
from sklearn.metrics import matthews_corrcoef, confusion_matrix, recall_score, accuracy_score
from scipy.special import expit as sigmoid


def train(train_iter, dev_iter,test_iter, model,class_weights,  args):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    best_acc = 0
    last_step = 0
    for epoch in range(1, args.epochs+1):
        for batch in train_iter:
            
            batch_size = batch[0].shape[0]
            structured=None
            if len(batch)==3:
                feature,structured, target = batch
                structured=torch.tensor(structured).float().cuda()
                
                
            else:
                feature, target = batch#batch.text, batch.label
#             feature.t_(), target.sub_(1)  # batch first, index align
            one_hot_target=F.one_hot(target, num_classes=2).float()
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()
                one_hot_target = one_hot_target.cuda()
                class_weights = class_weights.cuda()


            optimizer.zero_grad()
            if len(batch)==3:
                logit = model(x=feature,structured_x=structured)
            else:
                logit = model(feature)
            
            
            loss = F.binary_cross_entropy_with_logits(logit, one_hot_target, weight=class_weights)
            loss.backward()
            optimizer.step()

            steps += 1
#             if steps % args.log_interval == 0:
#                 corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
#                 accuracy = 100.0 * corrects/batch_size
#                 sys.stdout.write(
#                     '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps, 
#                                                                              loss.item(), 
#                                                                              accuracy.item(),
#                                                                              corrects.item(),
#                                                                              batch_size))
#             if steps % args.test_interval == 0:
        dev_acc = eval(dev_iter,test_iter, model,steps, args)
        if dev_acc > best_acc:
            best_acc = dev_acc
            last_step = steps
            if args.save_best:
                save(model, args.save_dir, 'best', steps)
#         else:
#             if steps - last_step >= args.early_stop:
#                 print('early stop by {} steps.'.format(args.early_stop))
#             elif steps % args.save_interval == 0:
#                 save(model, args.save_dir, 'snapshot', steps)


def eval(val_iter,test_iter, model,steps, args):
    model.eval()
    metrics_dict = {}
    for name, data_iter in {'val_':val_iter, 'test_':test_iter}.items():
        corrects, avg_loss = 0, 0
        all_preds = []
        all_targets = []
        all_logits = []
        for batch in data_iter:
            if len(batch)==3:
                feature,structured, target = batch
                structured = torch.tensor(structured).float().cuda()
            else:
                feature, target = batch#.text, batch.label
    #         feature.t_(), target.sub_(1)  # batch first, index align
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            if len(batch)==3:
                logit = model(x=feature,structured_x=structured)
            else:
                logit = model(feature)            
            loss = F.cross_entropy(logit, target, size_average=False)

            avg_loss += loss.item()
            
            
            preds = torch.max(logit, 1)[1].view(target.size()).data
            all_logits+=logit.tolist()
            all_preds += preds.tolist()
            all_targets += target.tolist()
            corrects += (preds == target.data).sum()

        size = len(data_iter.dataset)
        avg_loss /= size
        accuracy = 100.0 * corrects/size
    #     print(all_preds)
    #     print('!!!',all_targets)
        metrics_dict[name+'mcc'] = matthews_corrcoef(all_targets, all_preds)
        probs=sigmoid(all_logits)
        all_targets_2d = [[1,0] if x==0 else [0,1] for x in all_targets]
        metrics_dict[name+'accuracy'] =  accuracy_score(all_targets, all_preds)
        # print(all_targets)
        # print(all_preds)
        metrics_dict[name+'auc_avg'] = roc_auc_score(all_targets_2d, probs)
        metrics_dict[name+'auprc_avg'] = average_precision_score(all_targets_2d, probs)
        metrics_dict[name+'auc_1'] = roc_auc_score(all_targets, probs[:,1])
        metrics_dict[name+'auprc_1'] = average_precision_score(all_targets, probs[:,1])
        
        class_report = classification_report(all_targets,all_preds,output_dict=True) 
        metrics_dict[name+'precision'] = class_report['1']['precision']
        metrics_dict[name+'recall'] = class_report['1']['recall']
        metrics_dict[name+'f1-score'] = class_report['1']['f1-score']
        metrics_dict[name+'support'] = class_report['1']['support']
        metrics_dict[name+'specificity'] = class_report['0']['recall']
        metrics_dict['steps'] = steps
    fn = args.save_dir.replace('/','_')+'_'+'metrics.csv'
    df_dict = {k:[v] for k,v in metrics_dict.items()}
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if fn not in os.listdir(args.save_dir):
        pd.DataFrame(df_dict).to_csv(args.save_dir+'/'+fn, mode='a',header=True)
    else:
        pd.DataFrame(df_dict).to_csv(args.save_dir+'/'+fn, mode='a',header=False)



    print('\nEvaluation - loss: {:.6f} Eval_mcc: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       metrics_dict['val_mcc'],
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))
    
    return metrics_dict['val_mcc']


def predict(text, model, text_field, label_field, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    # text = text_field.tokenize(text)
#     text = text_field.preprocess(text)
#     text = [[text_field.vocab.stoi[x] for x in text]]
    x = torch.tensor(text)
    x = autograd.Variable(x)
    if cuda_flag:
        x = x.cuda()
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    return label_feild.vocab.itos[predicted.item()+1]


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
