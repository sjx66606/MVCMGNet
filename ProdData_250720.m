% 处理OA数据
clc;
clear;
eeglab;

path = 'H:\OA\OA-Data2\*OA*';
% 获取路径下的所有条目信息
subPath = dir(path);
% 过滤掉.和..
subPath = subPath(~ismember({subPath.name}, {'.', '..'}));
markers = { '21','31','41','51','23','33','43','53'};
timelist = [-2 5];

% 原始文件路径列表
filePaths = {

    'H:\OA\ProdData\OA-Data0720\20250108153604_薛希杰OA_MI右手11-1+42_ICAsub.mat',
    'H:\OA\ProdData\OA-Data0720\20250115143847_任贺OA-MI右手13-1+64_ICAsub.mat',
    'H:\OA\ProdData\OA-Data0720\20250307110015_孙敬一OA-MI左手30-1+33_ICAsub.mat',
    'H:\OA\ProdData\OA-Data0720\20250311152226_阎艳艳OA-MI右手35-1+83_ICAsub.mat',
    'H:\OA\ProdData\OA-Data0720\20250403140727_马海臣OA-FES右手8-1+15_ICAsub.mat',
    'H:\OA\ProdData\OA-Data0720\20250409114500_翟文宣OA-FES左手10-1+23_ICAsub.mat',
    'H:\OA\ProdData\OA-Data0720\20250409134010_曲廷波OA-FES左手11-1+36_ICAsub.mat',
    'H:\OA\ProdData\OA-Data0720\20250423161516_林政OA-FES13-1+19_ICAsub.mat',
    'H:\OA\ProdData\OA-Data0720\20250512090938_张保霖OA-FES左手16-1+42_ICAsub.mat'
};


% 获取所有文件名（去掉路径）
fileNames = cellfun(@(x) extractAfter(x, find(x == '\', 1, 'last')), filePaths, 'UniformOutput', false);
% 去掉文件名中 '+' 及其后面的内容（包括扩展名）
cleanNames = cellfun(@(x) regexprep(x, '\+.*$', ''), fileNames, 'UniformOutput', false);

% 提取匹配的文件（只保留在fileNames中存在的）
matchedFiles = subPath(ismember({subPath.name}, cleanNames));
subPath = matchedFiles;

for i = 1:length(subPath)

    % 获取脑电数据文件路径
    folderPath = strcat(subPath(i).folder,'\', subPath(i).name);
    % 读取脑电数据中的所有文件夹
    dataPaths = dir(folderPath);
    % 提取某个被试的所有文件夹
    names = {dataPaths.name};
    % 然后将其进行排序  后转换为字符形式
    % foldernames = sort_folders(folderPath, names);
    EEG.etc.eeglabvers = '2023.1';
    % 读取文件夹中的EEG数据
    [EEG, command] = pop_importNDF(folderPath);
    EEG = eeg_checkset( EEG );
    EEG=pop_chanedit(EEG, 'lookup','D:\\matlab tools\\eeglab2021.1\\plugins\\dipfit\\standard_BEM\\elec\\standard_1005.elc');
    EEG = eeg_checkset( EEG );
    % 取双侧乳突作为参考
    EEG = pop_reref(EEG, [17 18] );
    % 全脑平均作为参考
    %EEG = pop_reref( EEG, []);
    EEG = eeg_checkset( EEG );
    EEG.urchanlocs = EEG.chanlocs; % 备份原始位置
    EEG = pop_eegfiltnew(EEG, 'locutoff',1,'plotfreqz',0);
    EEG = eeg_checkset( EEG );
    EEG = pop_eegfiltnew(EEG, 'hicutoff',40,'plotfreqz',0);
    EEG = eeg_checkset( EEG );
    EEG = pop_eegfiltnew(EEG, 'locutoff',48,'hicutoff',52,'revfilt',1,'plotfreqz',0);
    EEG = eeg_checkset( EEG );
    % EEG = pop_rmbase( EEG, [],[]);
    % EEG = eeg_checkset( EEG );
    % pop_eegplot( EEG, 1, 1, 1);
    
    % 分段
    EEG = pop_epoch( EEG, markers, timelist, 'epochinfo', 'yes');
    EEG = eeg_checkset( EEG );
    % 基线矫正
    EEG = pop_rmbase( EEG, [timelist(1)*1000 0] ,[]);
    EEG = eeg_checkset( EEG );
    
    % 将原始数据中的21+闪烁的部分改为 22  这个更新后不保留
%     for k = 41:80
%         if length(EEG.epoch(k).eventtype) > 2 &&  EEG.epoch(k).eventtype{1}(2) == '1'
%             EEG.epoch(k).eventtype{1}(2) = '2';
%         else
%             disp('存在一些问题。。。。')
%         end
%     end
%     EEG = eeg_checkset( EEG );
    % 对event进行修改再更新epoch
     for k = 41:length(EEG.event)-1
        if EEG.event(k+1).type(2) == '0'  &&  EEG.event(k).type(2) == '1'
            EEG.event(k).type(2) = '2';
        end
    end
    EEG = eeg_checkset( EEG, 'eventconsistency');
    
    [orEEG, badchans] = pop_rejchan(EEG, 'elec',[1:30] ,'threshold', 5, 'measure', 'kurt','norm','on');
    % EEG = eeg_checkset( EEG );
    % 对坏导进行插值
    EEG = eeg_interp(EEG, badchans, 'spherical');
    EEG = eeg_checkset( EEG );
    
    % 检查坏导 坏段
    data = EEG.data;
    del_epoch = [];
    [channels, times, nepoch] = size(data);
    % 遍历每个矩阵并检查条件
    for k = 1:nepoch
        A = data(:,:,k);  % 当前矩阵
        if max(A(:)) > 1200 || min(A(:)) < -1200
            del_epoch = [del_epoch, k];  % 记录满足条件的矩阵序号
            greater_than_one = A > 300;
            count = sum(greater_than_one(:));
            A(greater_than_one) = 300;
            greater_than_one = A < -800;
            count = sum(greater_than_one(:));
            A(greater_than_one) = -300;
        end
        data(:,:,k) = A;
    end
    EEG.data = data;
    % 剔除坏段  数值为第几段
    EEG = pop_rejepoch( EEG, del_epoch ,0);
    
    
    
    % run ICA 
    dataRank = sum(eig(cov(double(EEG.data(:,:,1)'))) > 1E-6); % 求出数据的rank
    EEG = pop_runica(EEG, 'icatype', 'runica', 'extended',1, 'pca',20,...
        'stop',1E-7, 'interrupt','on'); % 排除眼电和非头皮上的参考电极
    EEG = pop_iclabel(EEG, 'default');
    EEG = eeg_checkset( EEG );
    EEG = pop_icflag(EEG, [0 0;0.9 1;0.9 1;0.9 1;0.9 1;0.9 1;0.9 1]);
    reject_idx = EEG.reject.gcompreject;
    reject_idx = reject_idx(1:20);   % 如果128导，只考虑前60个成分 如果64导，只考虑前30个成分  如果是32导，只考虑前20个成分 改成20
    reject_idx = find(reject_idx > 0);
    EEG = pop_subcomp(EEG, reject_idx);
    EEG = eeg_checkset( EEG );
    EEG = eeg_checkset( EEG );
    
    
    save(strcat('H:\OA\ProdData\OA-Data0720\',subPath(i).name,'+',string(length(del_epoch)),'_ICAsub'),'EEG','-v7.3');   
end
