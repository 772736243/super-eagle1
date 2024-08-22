#加载模块
import seaborn as sns
import pandas
#导入数据
tips = pandas.read_csv('E:\PytorchPro\SuperYOLO-main\\tips1.csv')
# tips = sns.load_dataset('tips1')
# sns.scatterplot(data=tips,x='total_bill',y='tip')
#size参数指定点的大小
# sns.scatterplot(data=tips,x='total_bill',y='tip',size='size')
#hue 按是否CSR进行分组

colors = ['#CC6666', '#6699CC'] #也可以直接['red', 'blue']
sns.scatterplot(data=tips,x='precision',y='cls_score',style='CSR',hue='CSR',palette=colors, size='CSR',sizes=[15,15])
# hue='CSR'表示颜色按照CSR分开，size='CSR'表示按照CSR区分大小,sizes=[15,15]表示CSR中不同类的大小，style='CSR'表示按照CSR中的类别区分形状
# sns.scatterplot(data=tips,x='recall',y='cls_precision',hue='smoker')
#保存图片
from matplotlib import pyplot as plt
# sns.scatterplot(data=tips,x='total_bill',y='tip',size='size',hue='smoker',style='time')
plt.savefig('scatterplot.pdf')