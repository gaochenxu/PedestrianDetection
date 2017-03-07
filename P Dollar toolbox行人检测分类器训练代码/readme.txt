需要在matlab中执行如下命令：
>>addpath(genpath('D:\toolbox'));savepath; %%此处目录为toolbox解压的目录
>>toolboxCompile; %%如果是32位电脑，需要编译，这一步可能问题会比较多，对计算机内部的c编译环境要求比较高。

%%如果编译出错，则执行(>>mex -setup)换一下编译器试一下
%%编译成功后可以运行demo测试