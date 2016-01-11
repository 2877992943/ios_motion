#!/usr/bin/env python
# encoding=utf-8

"""
log.tracer database
choose riding data ,sort,fea11,visual ,choose valid data
generate fea [50,11]
test with wb_para


"""

c_list=['running','walking','riding','sitting','driving']

dataPath='/home/yr/ios_motion/tmpData0110/' 
test_pack=11020#01234  #012 not sure walk   |mod<15 rid  | mod>=15 walkrun|but differ between people
para_pack=0#1with liangbin riding into trainset 
import numpy as np
import pylab as plt
import cPickle,math,random,theano
import theano.tensor as T
 
 
import time,scipy


deviceId={'huawei':'ffffffff-c7a8-3cd1-ffff-ffffea16e571', 
	'xiaomi':'ffffffff-c7a7-c7d4-0000-000031a92fbe', 
	'mine':'ffffffff-c95d-196f-0000-00007e282437',
	'chenyu':'ffffffff-c43f-692d-ffff-fffff4d57110',
	'chenyu_tx':'ffffffff-c28a-1bf0-0000-00005e774605',
	'chenyuIOS':'1D15E4FF-9C0D-4A06-91F1-80F5075EF4C8',
	'hengyIOS':'1A3C56C3-D5CC-4207-AF6E-109B5DE7F7FA',
	'libo':'CDE6342A-37ED-447F-9CC7-E1DD50BA49AD'
	}
sensor_list=["magneticSensor","accelerometer","proximity"]
inst_id_chenyu_IOS='yg80UzhxQFPPcwicTtU4lDTyP9IcqDs2'
inst_id_libo_IOS='G93jbBRhtiWVGFULyLR9gPHDjUkl3S3R'
inst_id_hengy_IOS='ElPG0i10dKplkcqkacurXVfwoyLHF3UV'
inst_id_hengy_IOS2='b71DnVu2vv2DQyTeNqrVEjNdnPN79NLR'
inst_id_js='m6qEd0WfJVrgxdcyFbXJdM6La6Gmf9He'
inst_id_zhangnn='L2OGFhzaHM2IbNzJ1NjEHceroElYVdk2'
inst_id_zhangnn1='GsjMrEReu1OC2wtkUpM5asbWEyJ5AdQ1'
inst_id_wanghy_ios='ozP8flfQnRoO2n0vQFQ3rShAuke4crQW'
inst_id_zhanggong='4Y5KKBtB7TuPrAiQd14xE1EarhJu0EQ0'
inst_id_mengxin='h8CQnD4g3VtojKIdymYEcRAIMidOH6wG'
inst_id_fengxp='aoXQRUGjNb25HyG8J3wfIB9APjWp6mOe'


year=2016
month=1;day_nov=11;
nov=[[month,day_nov,10,0,month,day_nov,13,0]]

c_list=['running','walking','riding','sitting','driving']
###########query label.data
device=deviceId['chenyuIOS'] 
sensor=sensor_list[0]
#duration=cy_rid_nov2
class_type=c_list[4]
#query log.tracer
period=nov[0] #5
inst_id=inst_id_zhanggong
##########

	


def save2pickle(c,name):
    write_file=open(dataPath+str(name),'wb')
    cPickle.dump(c,write_file,-1)#[ (timestamp,[motion,x,y,z]),...]
    write_file.close()
 
def load_pickle(path_i):
    f=open(path_i,'rb')
    data=cPickle.load(f)#[ [time,[xyz],y] ,[],[]...]
    f.close()
    #print data.__len__(),data[0]
    return data	


def fea4(obs):#[50,]obs
	#4
	mean=np.mean(obs);std=np.std(obs)
	min_i=np.min(obs);max_i=np.max(obs)
	f=np.array([mean,std,min_i,max_i])#[4,]
	dim=obs.shape[0]
	#percentile 5
	percentile=[10/100.*dim,25/100.*dim,50/100.*dim,75/100.*dim,90/100.*dim];#print percentile
	perc=[int(i) for i in percentile];#print perc
	obs_sort=np.sort(obs)#[50,]
	perc_i=obs_sort[perc];#print perc_i#[5,]
	gap4=perc_i[3]-perc_i[1]
	gap5=perc_i[4]-perc_i[0]
	#sum, square-sum 12
	position=[5,10,25,75,90,95]
	pos=[int(i/100.*dim) for i in position];#print pos
	sum_i=[np.sum(obs_sort[:i]) for i in pos]#[5,]
	sqrt_sum_i=[np.sqrt(np.dot(obs_sort[:i],obs_sort[:i])) for i in pos]
	#
	fea_i=np.concatenate((f,perc_i,gap4.reshape((1,)),gap5.reshape((1,)) ),axis=0);#print fea_i.shape
	return fea_i[:]#[10,]







def str2num(activity):#list
	type_list=['cycling', 'walking', 'stationary', 'running', 'automotive']
	num_list=[]
	for act in activity:
		for ind in range(type_list.__len__()):
			if act.find(type_list[ind])!=-1:
				num_list.append(ind)

			
	return np.array(num_list)

 




def mid_smooth(mod,wind_sz):
	if mod.shape[0]<=1:return mod
	else:
		#wind_sz=10
		n=mod.shape[0];print 'n',n
		mod1=[]
		for i in range(n)[:-wind_sz]:
			patch=mod[i:i+wind_sz]#[3,]
			pi=np.sort(patch)[int(wind_sz/2)]
			mod1.append(pi)
	#
		patch=mod[-wind_sz:]
		pi=np.sort(patch)[1]
		for t in range(wind_sz):
			mod1.append(pi)
	##
	 
		print np.array(mod1).shape[0]
		return np.array(mod1)
		
	 
	

def calcMod(xyz):#[n,3]
	return np.sqrt( (xyz*xyz).sum(axis=1) )#[n,]

		
 
def split_obs(obs_fea4): #[100,]->[3,50]
	return [obs_fea4[0:50],obs_fea4[25:75],obs_fea4[50:]  ]


def voteEvery3(y_pred):#[51,]->[17,]
	def majorityVote(arr):#[3,]->1x1
		uniq=np.unique(arr)
		if uniq.shape[0]==3 or uniq.shape[0]==1:return uniq[0]
		elif uniq.shape[0]==2:#for example [3,1,1]
			vote1=len(np.where(arr==uniq[0])[0])
			vote2=len(np.where(arr==uniq[1])[0])
			return [uniq[0] if vote1>vote2 else uniq[1] ][0]
		
		


	##############
	ker=3.;stri=ker
	 
	pred_list=[]; 
	num=int( (y_pred.shape[0]-ker)/stri ) +1
    	for i in range(num)[:]: #
        	vote=y_pred[stri*i:i*stri+ker]#[3,]
		if vote.shape[0]==ker: #[3,]
			vote=majorityVote(vote)#[3,]->1x1
			pred_list.append(vote)
	return pred_list
	
	
	
	
		
	
	
	
	

############3
if __name__=="__main__":
	
	  
	 
	 
 
	


	  
	#################
	#generate xy
	#########################
	xyz_acc=load_pickle(dataPath+'test-xyz-acc-ios-'+class_type)#[n,3]
	#xyz_acc=xyz_acc[:100,:]
	#
	assert isinstance(xyz_acc,np.ndarray) 
	assert xyz_acc.shape[0]>=100 
	assert xyz_acc.shape[1]==3
	#
	mod=calcMod(xyz_acc)
	mod_smooth3=mid_smooth(mod,3)

	###
	fea=mod_smooth3;print 'fea',fea.shape
	#kernel_sz=50.;stride=kernel_sz; #when training,cluster denoise windowSize=50
	kernel_sz=100.;stride=kernel_sz;
	obs_list=[]; 
	num=int( (fea.shape[0]-kernel_sz)/stride ) +1
    	for i in range(num)[:]: #[0,...100] total 101 
        	obs100=fea[stride*i:i*stride+kernel_sz]#[100,]
		if obs100.shape[0]==kernel_sz: #[100,]
			obs3=split_obs(obs100)#[100,]->[ [50,],[50,],[50,]  ]  overlap 3obs
			obs3=[ fea4(obs_i) for obs_i in obs3 ]#[ [11,],[11,],[11,] ] 3obs
			obs_list=obs_list+obs3# [11,]
			
	x_arr=np.array(obs_list);print 'x tobe tested',x_arr.shape
	 
	########
	#load model
	#########
	class_dic={'walking':0,'driving':1,'sitting':2,'unknown':3}#unknow:ride run   class_dic.keys() no sequence
	bdt=load_pickle(dataPath+'bdt')
	 
	y_pred=bdt.predict(x_arr);print 'pred',y_pred.shape
	y_pred=voteEvery3(y_pred)#[51,]->[17,]
	plt.figure();plt.subplot(211);plt.title('me_predict:walk0 drive1 sit 2 unknow3')
	plt.plot(y_pred,'bo')
	#activity_ios=load_pickle(dataPath+'activity_ios');print 'ios activity',activity_ios.shape
	#plt.subplot(212);plt.title('ios_predict:ride0, walking1, sit2, run3, drive4')
	#plt.plot(activity_ios,'bo')
	#####numlabel->string
	str_list=[]
	for yi in y_pred:
		for k,v in class_dic.items():
			if v==yi:
				yiStr=k;
				str_list.append(yiStr)
	print str_list
		
		
	 
	  


	
	plt.show()
	
	
	
	 

	

	 
	 
	 









	 

	 
	 
	 


 	
		
    
 
	
		
	
   		 



