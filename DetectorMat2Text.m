fid = fopen('ClassifierOut.txt','A');
fprintf(fid,'%d',detector.opts.nWeak);
fprintf(fid, '\r\n');
for i=1:detector.opts.nWeak
    for j=1:7
        fprintf(fid,'%d ',detector.clf.fids(j,i));
    end
    fprintf(fid, '\r\n');
end
for i=1:detector.opts.nWeak
    for j=1:7
        fprintf(fid,'%f ',detector.clf.thrs(j,i));
    end
    fprintf(fid, '\r\n');
end
for i=1:detector.opts.nWeak
    for j=1:7
        fprintf(fid,'%f ',detector.clf.hs(j,i));
    end
    fprintf(fid, '\r\n');
end
fclose(fid);

% fid=fopen('FeaOut.txt','A');
% for i=1:10
%     for j=1:198
%         for k=1:271
%             fprintf(fid, '%f ', P.data{1}(j,k,i));
%         end
%         fprintf(fid, '\r\n');
%     end
% end
% fclose(fid);