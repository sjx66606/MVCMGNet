% 处理长期数据
clc;
clear;
eeglab;


path = 'H:\OA\MI长期_sel\**';
% 获取路径下的所有条目信息
subPath = dir(path);
% 过滤掉.和..
subPath = subPath(~ismember({subPath.name}, {'.', '..'}));
save_folder = 'H:\OA\ProdData\MI(长期)_sel\EMG\';


for i = 1:length(subPath)

    % 获取脑电数据文件路径
    folderPath = strcat(subPath(i).folder,'\', subPath(i).name);
    % 读取脑电数据中的所有文件夹
    dataPaths = dir(folderPath);
    % 提取某个被试的所有文件夹
    names = {dataPaths.name};
    % 然后将其进行排序  后转换为字符形式
    % foldernames = sort_folders(folderPath, names);
    % 读取文件夹中的EEG数据
    [EMG, command] = pop_importNDF(folderPath);
    
    % 调用 getmain 函数处理 EEG 变量
    result = getEMGdat(EMG.data, 1000); 
    % 保存处理结果为文件名 + '_1.mat'
    name = subPath(i).name;
    newName = strrep(name, '_1_datRaw', '_1_Proced');
     saveName = fullfile(save_folder, [newName, '.mat']);
    save(saveName, 'result');
end