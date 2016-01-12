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
import lasagne
from leancloud import Object
from leancloud import Query
import leancloud
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
month=1;day_nov=12;
nov=[[month,day_nov,13,0,month,day_nov,14,0]]

c_list=['running','walking','riding','sitting','driving']
###########query label.data
device=deviceId['chenyuIOS'] 
sensor=sensor_list[0]
#duration=cy_rid_nov2
class_type=c_list[3]
#query log.tracer
period=nov[0] #5
inst_id=inst_id_zhanggong
##########
def connect_db_log():##sensor.log.tracer   not label.sensor, 
	import leancloud,cPickle
	appid = "9ra69chz8rbbl77mlplnl4l2pxyaclm612khhytztl8b1f9o"
	   
	appkey = "1zohz2ihxp9dhqamhfpeaer8nh1ewqd9uephe9ztvkka544b"
	#appid = "ckjjqzf3jqwl8k1u4o10j2hqp0rm0q6ferqcfb0xpw00e8zl"
	#appkey = "rn0qw8ib96xl0km63kylo6v5afuxclc8jti5ol8rx4kylqob"
	leancloud.init(appid, appkey)


def generate_stamp(period):
	#[8,28,8,33]->[(2015, 10, 20, 22, 30, 0, 0, 0, 0),(2015, 10, 20, 22, 48, 0, 0, 0, 0)]->stamp
	dur= [(year, period[0], period[1], period[2], period[3], 0, 0, 0, 0),\
		(year, period[4], period[5], period[6], period[7], 0, 0, 0, 0)]
	stamp_range0=[time2stamp(dur[0]),time2stamp(dur[1])]
	stamp_range=[t*1000 for t in stamp_range0]
	return stamp_range 

def time2stamp(t):
	#t = (2015, 9, 28, 12, 36, 38, 0, 0, 0)
	stamp = int(time.mktime( t )) ;
	return stamp
def connect_db():#label.data
	import leancloud,cPickle
	appid = "ckjjqzf3jqwl8k1u4o10j2hqp0rm0q6ferqcfb0xpw00e8zl"
	appkey = "rn0qw8ib96xl0km63kylo6v5afuxclc8jti5ol8rx4kylqob"
	leancloud.init(appid, appkey)

 
 


#####################
def get_content(results):#result is from find() 
 	 
	obs={}; 
	r=results
	for i in range(1):
		#print type(r.get("events")) 
		if len(r.get("events"))>=1:
			 
			print r.get("motion"),r.get("events").__len__()
			ll=r.get("events") #ll=[ {},{}...]
			for dic in ll[:]:#dic={timestamp:xxxx,value:[1,2,3]...}
			
			#print dic["timestamp"],' ',dic["values"][0],' ',dic["values"][1],' ',dic["values"][2]
				if dic["timestamp"] not in obs.keys():
					obs[ dic["timestamp"] ]=[r.get("motion"),\
					dic["values"][0],dic["values"][1],dic["values"][2]  ]
				###data form: {timestamp:[obs],...}  [obs]=[motion,x,y,z]
		 
	###########################
	"""
	for k,v in obs.items():
		print k,' ',v
	"""
	print 'final',obs.__len__()

	

	#print 'i',i,count  #query-has-limit100,real-count=320
	###################3
	return obs 
	
	 


def get_all(query,skip,result):
	limit=500
	query.limit(limit)
	query.skip(skip)
	found=query.find()
	if found and len(found)>0:
		result.extend(found)
		print 'av_utils get all,now result len:',len(result),'skip',skip
		return get_all(query,skip+limit,result)
	else:
		return result
	


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
		n=mod.shape[0];#print 'n',n
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
	
	
def replaceOver1000(vec):#[n,] 
	over1000_ind=np.where(vec>=1000)[0]
	#
	normal_ind=np.where(vec<1000)[0]
	mean_normal=np.mean(vec[normal_ind] )
	vec[over1000_ind]=mean_normal
	return vec	
	
		

def record2array(obj):#each record
	ll=obj.get('value');#print 'll',ll None 
	if ll!=None and 'events' in ll:
		events_list=ll["events"];#[{},{}...] {}={time value sensor}
		data_dic={'gyro':{},'acc':{},'orientation':{},'activity':{}}
		for obj_i in events_list:
			 
			sensor_i=obj_i.get('sensorName');#print sensor_i,obj_i.get('timestamp'),obj_i.get('values')
			data_dic[sensor_i][obj_i.get('timestamp')]=obj_i.get('values')
		#####sort by timestamp
		data_dic1={}
		for sensor,v in data_dic.items():
			if sensor in ['acc','orientation']:
				print 'sensor',sensor,v.__len__()  #{time:[xyz],...]
				if len(v)>1:
					ll=sorted(v.items(),key=lambda f:f[0],reverse=False)
					# # DATA FORMATE  {timestamp:[x y z],...}  ->  [ (timestamp,[x,y,z]),...]
					xyz=np.array([obs[1] for obs in ll]);#print '0',xyz.shape#[n,3]
					data_dic1[sensor]=xyz;#print np.sum(data_dic1['acc'])
				else: data_dic1[sensor]=np.zeros((10,3))
		####
		 
		if np.sum(data_dic1['acc'])!=0:
			#print 'i',data_dic1['acc'].shape
			return [data_dic1['acc'],data_dic1['orientation']]
		else:return [0,0]
		
		

def generateXY(xyz_acc,xyz_mag):
	mod_acc=calcMod(xyz_acc);mod_mag=calcMod(xyz_mag)
	#
	mod_acc_smooth3=mid_smooth(mod_acc,3);mod_mag_smooth3=mid_smooth(mod_mag,3);
	mod_mag_smooth3=replaceOver1000(mod_mag_smooth3)

	###acc
	fea=mod_acc_smooth3;print 'fea acc',fea.shape
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
			
	x_arr_acc=np.array(obs_list);print 'x tobe tested',x_arr_acc.shape

	###mag
	fea=mod_mag_smooth3;print 'fea mag',fea.shape
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
			
	x_arr_mag=np.array(obs_list);print 'x tobe tested',x_arr_mag.shape#[3,50]
	return x_arr_acc,x_arr_mag


def load_model():
	########
	#load model
	#########
	class_dic={'walking':0,'driving':1,'sitting':2,'unknown':3}#unknow:ride run   class_dic.keys() no sequence
	bdt=load_pickle(dataPath+'bdt0112')
	bdt_mag=load_pickle(dataPath+'mag-bdt0112')
	return [bdt,bdt_mag]
	
			 
		
def predictObs(x_arr_acc,x_arr_mag,bdt,bdt_mag):#[n,11] [n,11]
	class_dic={'walking':0,'driving':1,'sitting':2,'unknown':3}
	#acc 
	y_pred=bdt.predict(x_arr_acc);print 'pred',y_pred.shape
	#mag for predicted as driveSit
	ind_1=np.where(y_pred==1 )[0]
	ind_2=np.where(y_pred==2 )[0]
	ind_12=np.concatenate((ind_1,ind_2))
	x_arr_mag2=x_arr_mag[ind_12,:]
	y_pred2=bdt_mag.predict(x_arr_mag2); 
	y_pred[ind_12]=y_pred2


	##vote every 3 obs
	y_pred=voteEvery3(y_pred)#[51,]->[17,]
	
	###
	#####numlabel->string
	str_list=[]
	for yi in y_pred:
		for k,v in class_dic.items():
			if v==yi:
				yiStr=k;
				str_list.append(yiStr)
	return str_list
	

############3
if __name__=="__main__":
	


	  
	####init
	connect_db_log()
	log = leancloud.Object.extend('Log')
	log_query = leancloud.Query(log)
	#print 'all',log_query.count()##error
	inst_query = leancloud.Query(leancloud.Installation)
	print 'install',inst_query.count()#2335
	inst = inst_query.equal_to('objectId', inst_id).find();#print '1',inst[0]   
	
	#
	stamp_range=generate_stamp(period);print stamp_range
	 
	###########
	#db log.tracer,query sensor
	#################
	log_query.equal_to('installation', inst[0]).equal_to("type",'sensor').\
	less_than("timestamp", stamp_range[1]).greater_than("timestamp",stamp_range[0])
	print 'record count sensor',log_query.count()
		 
	######get all
	arr_list=[]
	if log_query.count()>0:
		record_list=get_all(log_query,0,[]);#print 'all',len(record_list),record_list[0]
		#instance object[<>,<>,<>,...]
		######each record
		
		for obj in record_list[:]:#each record
			#print '1',obj
			
			xyz_acc,xyz_mag=record2array(obj);#print '11',len(xyz_acc)
			if isinstance(xyz_acc,np.ndarray):
				arr_list.append([xyz_acc,xyz_mag])
				#print '1',xyz_acc.shape,len(arr_list)


		#print 'data dic',data_dic['acc'].__len__(),data_dic['acc'].keys()[0] 
	###########
	#db log.tracer,query motionLog
	#################
	log_query.equal_to('installation', inst[0]).equal_to("type",'motionLog').\
	less_than("timestamp", stamp_range[1]).greater_than("timestamp",stamp_range[0])
	print 'record count motionlog',log_query.count()
	######get all
	#arr_list=[]
	if log_query.count()>0:
		record_list=get_all(log_query,0,[]);#print 'all',len(record_list),record_list[0]
		#instance object[<>,<>,<>,...]
		######each record
		
		for i in range(len(record_list[:])):
			 
			obj=record_list[i]
			xyz_acc,xyz_mag=record2array(obj); #array100x3  or int0
			if isinstance(xyz_acc,np.ndarray):
				arr_list.append([xyz_acc,xyz_mag])
				print '1',xyz_acc.shape,len(arr_list)
				record_obj=obj
		
	

	 	
	 
	 

	 
	  

 	 
	######################3
	#clean data
	########################
	print 'start...'
	xyz_acc,xyz_mag=record2array(record_obj)#1 record object
	assert isinstance(xyz_acc,np.ndarray) 
	assert xyz_acc.shape[0]>=100 
	assert xyz_acc.shape[1]==3
	#print ' records',arr_list.__len__()
	bdt,bdt_mag=load_model()
	#for obs in arr_list:#[accxyz,magxyz]
	#xyz_acc,xyz_mag=obs #[100,3]
	
	x_arr_acc,x_arr_mag=generateXY(xyz_acc,xyz_mag)#[100,3]->[3,11]
	pred_list=predictObs(x_arr_acc,x_arr_mag,bdt,bdt_mag)
	print pred_list
	 
		 
	#
	 
	 
	
	 


	  
	
	 
	
	
	
	 

	

	 
	 
	 









	 

	 
	 
	 


 	
		
    
 
	
		
	
   		 



