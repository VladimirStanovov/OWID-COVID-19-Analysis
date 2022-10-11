import pip

def install(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        pip._internal.main(['install', package])

#install('numpy')
#install('pandas')
#install('scipy')
#install('mpi4py')
#install('openpyxl')
import numpy as np
import pandas as pd
import math
from scipy.integrate import odeint
from mpi4py import MPI

comm = MPI.COMM_WORLD
world_rank = comm.Get_rank()
world_size = comm.Get_size()

print(world_rank)
print("world_size",world_size)

#exit()

def MannWhitneyU_my(Sample1,Sample2):        
    NewSample = np.concatenate((Sample1,Sample2),axis=0)
    #print(NewSample)
    NewRanks, Groups = get_fract_ranks_and_groups(NewSample)
    #print(NewRanks)
    SumRanks = 0
    SumRanks2 = 0
    for i in range(Sample1.shape[0]):
        SumRanks += NewRanks[i]
        SumRanks2 += NewRanks[Sample1.shape[0]+i]
    #print(SumRanks)
    #print(SumRanks2)
    U1 = SumRanks - Sample1.shape[0]*(Sample1.shape[0]+1.0)/2.0
    U2 = SumRanks2 - Sample2.shape[0]*(Sample2.shape[0]+1.0)/2.0
    #print(U1,U2)
    #print(U1+U2)
    Umean = Sample1.shape[0]*Sample2.shape[0]/2.0
    #Ucorr = math.sqrt(Sample1.shape[0]*Sample2.shape[0]*(Sample1.shape[0]+Sample2.shape[0]+1.0)/12.0)
    GroupsSum = 0
    for index in Groups:
        GroupsSum += (index*index*index - index)/12
    N = Sample1.shape[0]+Sample2.shape[0]
    part1 = Sample1.shape[0]*Sample2.shape[0]/(N*(N-1.0))
    part2 = (N*N*N-N)/12.0
    Ucorr2 = math.sqrt(part1*(part2-GroupsSum))
    if(Ucorr2 == 0):
        return (0,0)
    Z1 = (U1 - Umean)/Ucorr2
    Z2 = (U2 - Umean)/Ucorr2    
    #print(Umean,Ucorr,Ucorr2)
    #print(Z1,Z2)
    #print(Z1+Z2)
    if(Z1 <= Z2):
        if(Z1 < -2.58):
            #print("worse")
            return (-1, Z1)
    else:
        if(Z2 < -2.58):   
            #print("better")
            return (1, Z1)
    #print("equal")
    return (0, Z1)

def MannWhitneyU_myZ(Sample1,Sample2):        
    NewSample = np.concatenate((Sample1,Sample2),axis=0)
    #print(NewSample)
    NewRanks, Groups = get_fract_ranks_and_groups(NewSample)
    #print(NewRanks)
    SumRanks = 0
    SumRanks2 = 0
    for i in range(Sample1.shape[0]):
        SumRanks += NewRanks[i]
        SumRanks2 += NewRanks[Sample1.shape[0]+i]
    #print(SumRanks)
    #print(SumRanks2)
    U1 = SumRanks - Sample1.shape[0]*(Sample1.shape[0]+1.0)/2.0
    U2 = SumRanks2 - Sample2.shape[0]*(Sample2.shape[0]+1.0)/2.0
    #print(U1,U2)
    #print(U1+U2)
    Umean = Sample1.shape[0]*Sample2.shape[0]/2.0
    #Ucorr = math.sqrt(Sample1.shape[0]*Sample2.shape[0]*(Sample1.shape[0]+Sample2.shape[0]+1.0)/12.0)
    GroupsSum = 0
    for index in Groups:
        GroupsSum += (index*index*index - index)/12
    N = Sample1.shape[0]+Sample2.shape[0]
    part1 = Sample1.shape[0]*Sample2.shape[0]/(N*(N-1.0))
    part2 = (N*N*N-N)/12.0
    Ucorr2 = math.sqrt(part1*(part2-GroupsSum))
    if(Ucorr2 != 0):
        Z1 = (U1 - Umean)/Ucorr2
        Z2 = (U2 - Umean)/Ucorr2
    else:
        return (0,0)
    #print(Umean,Ucorr,Ucorr2)
    #print(Z1,Z2)
    #print(Z1+Z2)
    if(Z1 <= Z2):
        if(Z1 < -2.58):
            #print("worse")
            return (-1, Z1)
    else:
        if(Z2 < -2.58):   
            #print("better")
            return (1, Z1)
    #print("equal")
    return (0, Z1)

def MannWhitneyU(Sample1,Sample2):        
    #print(Sample1,Sample2)
    counter = 0
    for i in range(Sample1.shape[0]):
        if Sample1[i] == Sample2[i]:
            counter += 1
    if counter != Sample1.shape[0]:
        [score,p] = mannwhitneyu(Sample1,Sample2)
    else:
        return 0
    #print(np.median(Sample1),np.median(Sample2))
    #print(score,p)
    sign = 0.01
    if p > sign:
        return 0
        #print("Equal")
    else:
        if p < sign and np.mean(Sample1) < np.mean(Sample2):
            #print("Worse")
            return -1
        elif np.mean(Sample1) > np.mean(Sample2):
            #print("Better")
            return 1
        else: 
            #print("Equal-2")
            return 0
def MannWhitneyUP(Sample1,Sample2):        
    #print(Sample1,Sample2)
    counter = 0
    for i in range(Sample1.shape[0]):
        if Sample1[i] == Sample2[i]:
            counter += 1
    if counter != Sample1.shape[0]:
        [score,p] = mannwhitneyu(Sample1,Sample2)
    else:
        return 0,0
    #print(np.median(Sample1),np.median(Sample2))
    #print(score,p)
    sign = 0.01
    if p > sign:
        return 0,p
        #print("Equal")
    else:
        if p < sign and np.mean(Sample1) < np.mean(Sample2):
            #print("Worse")
            return -1,p
        elif np.mean(Sample1) > np.mean(Sample2):
            #print("Better")
            return 1,p
        else: 
            #print("Equal-2")
            return 0,p
def ranks(ranking):
    return list(ranking.ranks())   
def get_fract_ranks(data):
    sort_index = np.argsort(-data)
    sort_list = -np.sort(-data)
    new_ranks = ranks(Ranking(sort_list, FRACTIONAL))
    index_rank = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        new_rank_inv = data.shape[0] - new_ranks[i] - 1
        index_rank[sort_index[i]] = new_rank_inv
    #print(sort_index)
    #print(sort_list)
    #print(new_ranks)
    #print(index_rank)
    return index_rank
def FriedmanSTest(ResultsFunction, func_num, size, NRuns):
    rankarray = np.zeros((NRuns,size))
    for i in range(NRuns):
        rankarray[i] = get_fract_ranks(np.transpose(ResultsFunction[:,i,func_num]))
    sumranks = np.zeros(size)
    avgranks = np.zeros(size)
    for i in range(size):
        sumranks[i] = np.sum(rankarray[:,i])
    avgranks = sumranks / NRuns
    raverage = (size+1)/2.0
    #print(raverage)
    FriedmanS = 0
    for i in range(size):
        FriedmanS += (avgranks[i] - raverage)*(avgranks[i] - raverage)
    FriedmanS *= 12*NRuns/(size*(size+1.))
    #print(FriedmanS)
    return avgranks


def get_fract_ranks_and_groups(data):
    sort_index = np.argsort(-data)
    #print(sort_index)
    sort_list = -np.sort(-data)
    #print(sort_list)
    groups = []
    my_new_ranks = np.zeros(data.shape[0])
    counter = 0
    while(True):
        if(counter == data.shape[0]):
            break
        if(counter == data.shape[0]-1):
            my_new_ranks[counter] = counter
            #print("break1")
            break
        if(sort_list[counter] != sort_list[counter+1]):
            my_new_ranks[counter] = counter
            #print("assigning ",my_new_ranks[counter])
            #print("counter = ",counter)
            counter+=1            
        else:
            avgrank = 0
            start = counter
            #print("start = ",start)
            while(sort_list[start] == sort_list[counter]):
                avgrank += counter
                #print("avg caclulating",counter)
                #print("counter = ",counter)
                counter+=1                
                if(counter == data.shape[0]):
                    #print("break2")
                    break
            #print("fin counter = ",counter)
            avgrank = avgrank / (counter - start)
            #print("avgrank = ",avgrank)
            groups.append(counter - start)
            for i in range(start,counter):
                my_new_ranks[i] = avgrank
                #print("assigning ",my_new_ranks[i])
    #print(my_new_ranks)
    #new_ranks = ranks(Ranking(sort_list, FRACTIONAL))
    index_rank = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        new_rank_inv = data.shape[0] - my_new_ranks[i]
        index_rank[sort_index[i]] = new_rank_inv   
    #print(new_ranks)
    #print(index_rank)
    #print(groups)
    #print(new_ranks - my_new_ranks)
    return index_rank, groups

#testdata = np.array([0.11,45,0.11,22,3,3,3,3,4,45,45,8,14])
#testdata = np.array([0.11,45,0.11,22,3,3,3,3,4,45,45,8,14])
#print(get_fract_ranks_and_groups(testdata)[1])

#Data = pd.read_excel("owid-covid-data_a_.xlsx")
Data = pd.read_excel("owid-covid-data_march_2021.xlsx")
use_SI_ISO = 0

Data_SI_A = pd.read_excel("results_04_SI_A.xlsx")
#Data.iloc[:,0]
SI_iso = Data_SI_A.iso_code_out.values
SI_start = Data_SI_A.start_of_wave_out.values
SI_end = Data_SI_A.end_of_wave_out.values
SI_err = Data_SI_A.relerrorValue_out.values
SI_par1 = Data_SI_A.infectionCoef_out.values
SI_par2 = Data_SI_A.percentOfSusceptible_out.values
#print(SI_iso)
#print(SI_start)
#print(SI_end)
#print(SI_err)
#print(SI_par1)
#print(SI_par2)
use_SI_ISO = 1

Data_SI_A = pd.read_excel("BD_results_04_2.xlsx")
#Data.iloc[:,0]
SI_iso = Data_SI_A.iso_code_out.values
SI_start = Data_SI_A.start_of_wave_out.values
SI_end = Data_SI_A.end_of_wave_out.values
SI_err = Data_SI_A.relerrorValue_out.values
SI_par1 = Data_SI_A.q_infectionCoef_imitation_from_others_out.values
SI_par2 = Data_SI_A.p_innovativity_from_air_out.values
SI_par3 = Data_SI_A.percentOfSusceptible_out.values
#print(SI_iso)
#print(SI_start)
#print(SI_end)
#print(SI_err)
#print(SI_par1)
#print(SI_par2)
#print(SI_par3)
use_SI_ISO = 1

NamesList = Data.iso_code.unique()
NamesList = NamesList[:len(NamesList)]
NamesList2 = Data.location.unique()
NamesList2 = NamesList2[:len(NamesList)]
PopulationSizes = np.zeros(len(NamesList))
for i in range(len(NamesList)):
    Df1 = Data[Data.iloc[:,0] == NamesList[i]].population
    Df1 = Df1.fillna(0)
    Cases1 = np.array(Df1)
    PopulationSizes[i] = np.array(Df1)[0]
    #print(NamesList[i],len(Cases1))#,PopulationSizes[i])
    #plt.plot(Pop1)
    #plt.title(NamesList2[i])
    #plt.show()
len(NamesList)

folder = "owid_graphs/"
newNCases = np.zeros(len(NamesList))
makegraphs = 0
MidPoints = np.zeros(len(NamesList))

total_n_c = 0

StartArr = np.zeros(len(NamesList))

for i in range(len(NamesList)):
#for i in range(11,12):
    Df1 = Data[Data.iloc[:,0] == NamesList[i]].total_cases
    Df1 = Df1.fillna(0)
    Cases1 = np.array(Df1)    
    
    SI_num = 0
    use = 1
    if(use_SI_ISO):
        for j in range(SI_iso.shape[0]):
            if(SI_iso[j] == NamesList[i]):
                use = 1
                SI_num = j
                break
    if(use == 0):
        continue        
        
    total_n_c += 1
        
    #print(NamesList2[i],NamesList[i])
    
    if(makegraphs and False):
        fig = plt.figure(figsize=(10, 10))    
        plt.plot(Cases1)
        plt.title(NamesList2[i])
        plt.ylabel("Total cases")
        plt.xlabel("Days")
        plt.show()
    #folder = "owid_graphs/"
    #fig.savefig(folder+"Curve_"+NamesList2[i]+".eps")
    #fig.savefig(folder+"Curve_"+NamesList2[i]+".png")
    Df1 = Data[Data.iloc[:,0] == NamesList[i]].new_cases_smoothed
    Df1 = Df1.fillna(0)
    Cases2 = np.array(Df1)
    
    if(makegraphs):
        fig = plt.figure(figsize=(7, 4))    
        plt.plot(Cases2,color=(0.1, 0.1, 0.9),lw=2, linestyle='-')
        plt.title(NamesList2[i])
        plt.ylabel("Cases per day")
        plt.xlabel("Days")    
        plt.grid(True)
        plt.show()
        fig.savefig(folder+"Sm_perday_"+NamesList2[i]+".eps")
        fig.savefig(folder+"Sm_perday_"+NamesList2[i]+".png")   
        
    Cases3 = np.zeros(Cases2.shape[0])
    for j in range(Cases3.shape[0]-1):
        Cases3[j] = Cases2[j+1]-Cases2[j]
    for j in range(1,Cases3.shape[0]):
        Cases3[j] = Cases3[j]*0.02+Cases3[j-1]*0.98
        
    if(makegraphs):        
        fig = plt.figure(figsize=(7, 4))       
        plt.plot(Cases3,color=(0.1, 0.1, 0.9),lw=2, linestyle='-')
        plt.title(NamesList2[i])    
        plt.ylabel("Smoothed derivative of cases per day")
        plt.xlabel("Days")    
        plt.grid(True)
        plt.show()
        fig.savefig(folder+"Sm_der_"+NamesList2[i]+".eps")
        fig.savefig(folder+"Sm_der_"+NamesList2[i]+".png") 
    
    waspositive = 0
    maxpos = Cases3[0]
    wasnegative = 0
    minneg = Cases3[0]
    lenpercent = 0.07
    
    hasnew = 0
    peakvalue = 0
    peakvalue2 = 0
    peakvalue2n = 0
    midsaved = -1
    for j in range(Cases3.shape[0]):
        if(Cases3[j] > maxpos):
            maxpos = Cases3[j]
        if(Cases3[j] < minneg):
            minneg = Cases3[j]
        if(Cases2[j] > peakvalue):
            peakvalue = Cases2[j]
        if(Cases3[j] > peakvalue2):
            peakvalue2 = Cases3[j]
        if(Cases3[j] < peakvalue2):
            peakvalue2n = Cases3[j]
        if(j > lenpercent*Cases3.shape[0] and Cases3[j] > 0 and Cases3[j-int(lenpercent*Cases3.shape[0])] > 0):
            waspositive = 1
        if(waspositive == 1 and Cases3[j] < 0 and midsaved==-1):
            midsaved = 1
            MidPoints[i] = j
        if(j > lenpercent*Cases3.shape[0] and Cases3[j] < 0 and Cases3[j-int(lenpercent*Cases3.shape[0])] < 0):
            wasnegative = 1
        #print(j,waspositive,wasnegative)
        if(waspositive == 1 and wasnegative == 1 and Cases3[j-1] < 0 and Cases3[j] > 0):            
            hasnew = 1
            newNCases[i] = j            
            break 
        if(waspositive == 1 and wasnegative == 1 and Cases2[j] < peakvalue*0.03):            
            hasnew = 1
            newNCases[i] = j            
            break 
    if(hasnew == 0):
        newNCases[i] = Cases3.shape[0]
    if(MidPoints[i] < 0.5*newNCases[i]):
        MidPoints[i] = 0.5*newNCases[i]
    if(MidPoints[i] > 0.75*newNCases[i]):
        MidPoints[i] = 0.75*newNCases[i]           
    
    
    #print(newNCases[i])
    endCases = newNCases[i]
    
    Gap = 150
    minStart = 0
    totalI = 0
    ndays = 3
    memdays = np.zeros(ndays)
    counter = 0
    smcs1 = np.zeros(Cases1.shape[0])
    for j in range(1,Cases1.shape[0]):
        smcs1[j] = smcs1[j-1]*0.8+Cases1[j]*0.2
    for j in range(Cases1.shape[0]):
        #print(i)
        if(j>0):
            if(smcs1[j] - smcs1[j-1] > 1):
                memdays[counter] = 1
            else:
                memdays[counter] = 0
            counter+= 1
            counter = counter%ndays
        
        if(sum(memdays) == ndays):
            minStart = j
            break
    
    if(minStart >= newNCases[i]):
        minStart = int(0.1*newNCases[i])
        
    StartArr[i] = minStart
    
    #print(minStart)
    #print(endCases)
    #print(endCases-minStart)
    
    #print(np.linspace(minStart,endCases,endCases-minStart))
    
    #newNCases[i] = endCases
    
    #print(minStart)
    #print(endCases)
    #print(endCases-minStart)
    #print
    #print(MidPoints[i],newNCases[i])   
    #print(minStart,newNCases[i])           
    
    if(makegraphs):
        fig = plt.figure(figsize=(10, 6))    
        plt.plot(Cases1,color=(0.1, 0.1, 0.9),lw=2, linestyle='-')
        plt.plot(np.linspace(minStart,int(newNCases[i]),int(newNCases[i])-minStart),Cases1[minStart:int(newNCases[i])],color=(0.9, 0.1, 0.1),lw=6, linestyle='-')
        plt.title(NamesList2[i])
        plt.ylabel("Total cases")
        plt.xlabel("Days")    
        plt.grid(True)
        plt.show()
        folder = "owid_graphs/"
        #fig.savefig(folder+"Curve_"+NamesList2[i]+".eps")
        #fig.savefig(folder+"Curve_"+NamesList2[i]+".png")    
        
    if(makegraphs):
        fig = plt.figure(figsize=(10, 6))    
        #plt.plot(Cases1,color=(0.1, 0.1, 0.9),lw=2, linestyle='-')
        plt.plot(Cases1[:int(newNCases[i])],color=(0.9, 0.1, 0.1),lw=6, linestyle='-')
        plt.title(NamesList2[i])
        plt.ylabel("Total cases")
        plt.xlabel("Days")    
        plt.grid(True)
        plt.show()
        folder = "owid_graphs/"
        #fig.savefig(folder+"Curve_"+NamesList2[i]+".eps")
        #fig.savefig(folder+"Curve_"+NamesList2[i]+".png") 
    
    #print(SI_start[SI_num],SI_end[SI_num])   
    if(makegraphs and False):
        fig = plt.figure(figsize=(10, 6))    
        plt.plot(Cases1,color=(0.1, 0.1, 0.9),lw=2, linestyle='-')
        plt.plot(np.linspace(SI_start[SI_num],SI_end[SI_num],SI_end[SI_num]-SI_start[SI_num]),Cases1[SI_start[SI_num]:SI_end[SI_num]],color=(0.1, 0.9, 0.1),lw=6, linestyle='-')
        plt.title(NamesList2[i])
        plt.ylabel("Cases per day")
        plt.xlabel("Days")    
        plt.grid(True)
        plt.show()
        
#print(total_n_c)


#newNCases = np.loadtxt("newNCases.txt")
#print(newNCases.shape)
#print(len(NamesList))
#print(newNCases)

TotalCases = np.zeros(newNCases.shape)
for i in range(len(NamesList)):
    Df1 = Data[Data.iloc[:,0] == NamesList[i]].total_cases
    Df1 = Df1.fillna(0)
    Cases1 = np.array(Df1)
    
    
    use = 0
    if(use_SI_ISO):
        for j in range(SI_iso.shape[0]):
            if(SI_iso[j] == NamesList[i]):
                use = 1
                break
    if(use == 0):
        continue    
    
    if(use_SI_ISO):
        for j in range(SI_iso.shape[0]):
            if(SI_iso[j] == NamesList[i]):
                newNCases[i] = SI_end[j]-1
                newNCases[i] = max(0,newNCases[i])
                #print("Changed2 "+SI_iso[j]+"\t"+(str(newNCases[j])))   
    
    
    #if(len(Cases1) != newNCases[i]):
    #    print(NamesList2[i]+"\tCHANGED\t"+str(len(Cases1))+"\t"+str(newNCases[i]))
    #else:
    #    print(NamesList2[i])
    #print(i)
    TotalCases[i] = np.max(Cases1[:int(newNCases[i])])
    #plt.plot(Cases1[:int(newNCases[i])])
    #plt.title(NamesList2[i])
    #plt.show()

CType = np.loadtxt("covid_curve_type.txt")
for i in range(len(NamesList)):
    if(NamesList2[i] == "Asia" or
       NamesList2[i] == "Africa" or
       NamesList2[i] == "North America" or    
       NamesList2[i] == "South America" or
       NamesList2[i] == "Europe" or
       NamesList2[i] == "Oceania" or
       NamesList2[i] == "China" or
       NamesList2[i] == "Denmark" or
       NamesList2[i] == "Brazil" or
       NamesList2[i] == "Oceania" or
       NamesList2[i] == "Uganda" or
       NamesList2[i] == "Vietnam" or
       NamesList2[i] == "Poland" or
       NamesList2[i] == "Northern Macedonia" or
       NamesList2[i] == "European Union"):
        CType[i] = -1
        #print(NamesList2[i])
#print(CType.shape)   

for i in range(len(NamesList)):
    Df1 = Data[Data.iloc[:,0] == NamesList[i]].date
    Df1.fillna(0)
    #print(NamesList2[i],StartArr[i],newNCases[i])
    #print(Df1)
    if(PopulationSizes[i] < 1000000):
        #print(NamesList2[i] + " less then 1m")
        continue
    if(CType[i] != 0):
        #print(NamesList2[i] + " not S-curve")
        continue
    
    Df2 = Data[Data.iloc[:,0] == NamesList[i]].total_cases
    Df2 = Df2.fillna(0)
    Cases1 = np.array(Df2)  
    CasesStart = Cases1[int(StartArr[i])]
    CasesEnd = Cases1[int(newNCases[i])-1]
    #print(str(CasesStart),str(CasesEnd))
    
    
    #print(NamesList2[i]+"\t"+Df1.iloc[int(StartArr[i])]+"\t"+str(int(CasesStart))+"\t"+Df1.iloc[int(newNCases[i])-1]+"\t"+str(int(CasesEnd))+"\t"+str(int(PopulationSizes[i])))    
	
	
def s_curve(steep,height,loc,x):
    return (1.0-1.0/(1.0+np.exp((x-loc)*steep)))*height
def MAE(Orig,Pred):
    if(Orig.shape[0] > 0):
        return sum(abs(Orig-Pred))/Orig.shape[0]
    return 100000000000000
def RMSE(Orig,Pred):
    if(Orig.shape[0] > 0):
        return sum((Orig-Pred)*(Orig-Pred))/Orig.shape[0]
    return 100000000000000
def RE(Orig,Pred):
    if(Orig.shape[0] > 0):
        return sum((Orig-Pred)*(Orig-Pred))/sum((Orig)*(Orig))
    return 100000000000000
def MAPE(Orig,Pred):
    if(Orig.shape[0] > 0):
        #print(Orig-Pred)
        #print(np.abs(Orig-Pred))
        #print(Orig)
        #print(Pred)
        #print(np.abs(Orig-Pred)/(Orig))
        #print(Orig.shape)                
        return np.sum(np.abs(Orig-Pred)/(Orig))/Orig.shape[0]
    return 100000000000000
def Rsquared(Orig,Pred):
    if(Orig.shape[0] > 0 and np.std(Orig) !=0 and np.std(Pred) != 0):
        return ((sum(Pred*Orig)/Orig.shape[0] - np.mean(Orig)*np.mean(Pred))/np.std(Orig)/np.std(Pred))*((sum(Pred*Orig)/Orig.shape[0] - np.mean(Orig)*np.mean(Pred))/np.std(Orig)/np.std(Pred))
    return 100000000000000
	
def tsi(y,t,betta,thetta,N,pop,omega):
    Ti=y[0]
    Si=y[1]    
    Ii=y[2]
    SIi=y[3]
    Ni=y[4]
    f0=-omega*Ti
    f1=-betta*Ii*Si/Ni - thetta*Si+omega*Ti              #dS/dt=-betta*I*S/N 
    f2= betta*Ii*Si/Ni + thetta*Si              #dI/dt=betta*I*S/N
    f3= betta*Ii*Si/Ni + thetta*Si
    f4=+omega*Ti
    return [f0,f1,f2,f3,f4]
    
def getTotalInfCurveTSI(betta,thetta,Tmax,height,start,pop,omega):
    T0 = pop
    S0 = height-start
    I0 = start
    SI0 = start
    y0 = [T0, S0, I0, SI0, height]                     # initial condition vector
    t  = np.linspace(0, Tmax-1, Tmax)          # time grid
    soln = odeint(tsi, y0, t, args=(betta,thetta,height,pop,omega),rtol=1e-10)
    T = soln[:, 0]
    S = soln[:, 1]
    I = soln[:, 2]
    SI =soln[:, 3]
    N = soln[:, 4]
    return SI


#TSI = getTotalInfCurveTSI(0.1,0.0,301,100001,1,100000,0e-5)
TSI = getTotalInfCurveTSI(0.05,0.00,301,100001*0.005,1,10000000,0e-5)
TSI_M=getTotalInfCurveTSI(0.05,0.01,301,100001*0.005,1,10000000,0e-5)

if(makegraphs):
    fig = plt.figure(figsize=(7, 4))
    plt.plot(TSI)
    plt.plot(TSI_M)
    plt.grid(True)
    plt.xlabel("Days")
    plt.ylabel("Total cases")
    #plt.ylim(0,100000)
    plt.legend(("SI","SI modified"))
    plt.show()
    folder = "owid_graphs/"
    fig.savefig(folder+"TSI_M_.eps")
    fig.savefig(folder+"TSI_M_.png") 
    #print(TSI.shape)
    np.savetxt("TSI.txt",TSI)
    np.savetxt("TSI_Mod.txt",TSI)
    #print(TSI[0])
    #print(TSI[-1])



def tbd(y,t,p,q,M,pop,omega):
    Ti=y[0]
    Mi=y[1]
    Ni=y[2]
    f0=-omega*Ti
    f1=-p*(Mi-Ni)-q*(1.0/Mi)*Ni*(Mi-Ni)+omega*Ti
    f2= p*(Mi-Ni)+q*(1.0/Mi)*Ni*(Mi-Ni)            #dNt/dt=p*(M-Nt)+q*(1.0/M)*Nt*(M-Nt)
    return [f0,f1,f2]
    
def getTotalInfCurveTBD(p,q,Tmax,height,start,pop,omega): 
    y0 = [pop,height,start]
    t  = np.linspace(0, Tmax-1, Tmax)  
    soln = odeint(tbd, y0, t, args=(p,q,height,pop,omega),rtol=1e-10)
    return soln[:,2]


BD =   getTotalInfCurveTBD(0.0001,0.1,151,100001,1,10000000,0e-5)

if(makegraphs):
    fig = plt.figure(figsize=(7, 4))
    plt.plot(BD)
    plt.grid(True)
    plt.xlabel("Days")
    plt.ylabel("Total cases")
    #plt.ylim(0,100000)
    plt.legend(("TBD model",""))
    plt.show()
    folder = "owid_graphs/"
    fig.savefig(folder+"TBD_M_.eps")
    fig.savefig(folder+"TBD_M_.png") 
    #print(BD.shape)
    np.savetxt("TBD.txt",BD)
    #print(BD[0])


def tsir(y,t,betta,gamma,thetta,N,pop,omega):
    Ti= y[0] 
    Si= y[1]    
    Ii= y[2]
    Ri= y[3]    
    SIi=y[4]
    Ni= y[5]
    f0=-Ti*omega
    f1=-betta*Ii*Si/Ni - thetta*Si + omega*Ti    #dS/dt=-betta*I*S/N 
    f2= betta*Ii*Si/Ni + thetta*Si - gamma*Ii    #dI/dt=betta*I*S/N-gamma*I
    f3= gamma*Ii                                #dR/dt=gamma*I    
    f4= betta*Ii*Si/Ni + thetta*Si 
    f5= Ti*omega 
    return [f0,f1,f2,f3,f4,f5]
    
def getTotalInfCurveTSIR(betta,gamma,thetta,Tmax,height,start,pop,omega):
    T0 = pop
    S0 = height-start
    I0 = start
    R0 = 0
    SI0 = start
    y0 = [T0, S0, I0, R0, SI0, height]                     # initial condition vector
    t  = np.linspace(0, Tmax-1, Tmax)              # time grid
    soln = odeint(tsir, y0, t, args=(betta,gamma,thetta,height,pop,omega),rtol=1e-10)
    T = soln[:, 0]
    S = soln[:, 1]
    I = soln[:, 2]
    R = soln[:, 3]
    SI =soln[:, 4]
    N =soln[:, 5]
    return SI

SI = getTotalInfCurveTSIR(0.1,0.01,0,301,100001,1,10000000,0e-5)
SI_M=getTotalInfCurveTSIR(0.06,0.01,0.0001,301,100001,1,10000000,0e-5)

if(makegraphs):
    fig = plt.figure(figsize=(7, 4))
    plt.plot(SI)
    plt.plot(SI_M)
    plt.grid(True)
    plt.xlabel("Days")
    plt.ylabel("Total cases")
    #plt.ylim(0,100000)
    plt.legend(("TSIR","Modified TSIR"))
    plt.show()
    folder = "owid_graphs/"
    fig.savefig(folder+"TSIR_M_.eps")
    fig.savefig(folder+"TSIR_M_.png") 
    #print(SI.shape)
    np.savetxt("TSIR.txt",SI)
    np.savetxt("TSIR_Mod.txt",SI_M)
    #print(SI[0])
    #print(SI[-1])


def tseir(y,t,alpha,betta,sigma,gamma,thetta,N,pop,omega):
    Ti= y[0]
    Si= y[1]
    Ei= y[2]
    Ii= y[3]
    Ri= y[4]
    SIi=y[5]
    Ni= y[6]
    
    f0=-Ti*omega
    f1=-betta*Ii*Si/Ni-alpha*Si*Ei/Ni - thetta*Si + Ti*omega        #dS/dt=-betta*I*S 
    f2= betta*Ii*Si/Ni+alpha*Si*Ei/Ni + thetta*Si - sigma*Ei        #dE/dt=betta*I*S-sigma*E
    f3=-gamma*Ii+sigma*Ei                                           #dI/dt=sigma*E-gamma*I
    f4= gamma*Ii                                                    #dR/dt=gamma*I
    f5=+sigma*Ei
    f6= Ti*omega 
    
    return [f0,f1,f2,f3,f4,f5,f6]  
    
def getTotalInfCurveTSEIR(alpha,betta,sigma,gamma,thetta,Tmax,height,start,pop,omega):
    T0 = pop
    S0 = height-start
    E0 = 0
    I0 = start
    R0 = 0
    SI0 = start
    y0 = [T0, S0, E0, I0, R0, SI0, height]      # initial condition vector
    t  = np.linspace(0, Tmax-1, Tmax)           # time grid

    soln = odeint(tseir, y0, t, args=(alpha,betta,sigma,gamma,thetta,height,pop,omega),rtol=1e-10)
    T = soln[:, 0]
    S = soln[:, 1]
    E = soln[:, 2]
    I = soln[:, 3]
    R = soln[:, 4]
    SI= soln[:, 5]
    N = soln[:, 6]
    
    return SI

SI =   getTotalInfCurveTSEIR(0.0,1.0,0.01, 0.0005,0.00, 301,100001,1,100000,0.00000)
SI_M = getTotalInfCurveTSEIR(0.0,1.0,0.006,0.0005,0.001,301,100001,1,100000,0.00000)

if(makegraphs):
    fig = plt.figure(figsize=(7, 4))
    plt.plot(SI)
    plt.plot(SI_M)
    plt.grid(True)
    plt.xlabel("Days")
    plt.ylabel("Total cases")
    #plt.ylim(0,100000)
    plt.legend(("SEIR","Modified SEIR"))
    plt.show()
    folder = "owid_graphs/"
    fig.savefig(folder+"TSEIR_M_.eps")
    fig.savefig(folder+"TSEIR_M_.png") 
    fig.savefig(folder+"TSEIR_M_.svg") 
    #print(SI.shape)
    np.savetxt("TSEIR.txt",SI)
    np.savetxt("TSEIR_Mod.txt",SI_M)


def tseiar(y,t,alpha,betta,sigma,gamma,thetta,N,pop,omega,kappa,mu):
    Ti= y[0]
    Si= y[1]
    Ei= y[2]
    Ii= y[3]
    Ai= y[4]
    Ri= y[5]
    SIi=y[6]
    SAi=y[7]
    Ni= y[8]
    
    f0=-Ti*omega
    f1=-betta*Ii*Si/Ni - mu*betta*Ai*Si/Ni - alpha*Si*Ei/Ni - thetta*Si + Ti*omega        #dS/dt=-betta*I*S-mu*betta*U*S
    f2= betta*Ii*Si/Ni + mu*betta*Ai*Si/Ni + alpha*Si*Ei/Ni + thetta*Si - sigma*Ei        #dE/dt= betta*I*S+mu*betta*U*S-sigma*E
    f3=-gamma*Ii+kappa*sigma*Ei                                                           #dI/dt=-gamma*I+alpha*sigma*E
    f4=-gamma*Ai+(1.0-kappa)*sigma*Ei                                                     #dA/dt=-gamma*U+(1-alpha)*sigma*E
    f5= gamma*(Ii+Ai)                                                                     #dR/dt= gamma*(I+U)
    f6=+kappa*sigma*Ei                                                                    
    f7=+(1.0-kappa)*sigma*Ei
    f8= Ti*omega 
    
    return [f0,f1,f2,f3,f4,f5,f6,f7,f8]  
    
def getTotalInfCurveTSEIAR(alpha,betta,sigma,gamma,thetta,Tmax,height,start,pop,omega,kappa,mu):
    T0 = pop
    S0 = height-start
    E0 = 0
    I0 = start
    A0 = start/kappa
    R0 = 0
    SI0 = start
    SA0 = 0
    y0 = [T0, S0, E0, I0, A0, R0, SI0, SA0, height]      # initial condition vector
    t  = np.linspace(0, Tmax-1, Tmax)           # time grid

    soln = odeint(tseiar, y0, t, args=(alpha,betta,sigma,gamma,thetta,height,pop,omega,kappa,mu),rtol=1e-10)
    T = soln[:, 0]
    S = soln[:, 1]
    E = soln[:, 2]
    I = soln[:, 3]
    A = soln[:, 4]
    R = soln[:, 5]
    SI= soln[:, 6]
    SA= soln[:, 7]
    N = soln[:, 8]
    
    return SI

def getTotalInfCurveTSEIAR_g(alpha,betta,sigma,gamma,thetta,Tmax,height,start,pop,omega,kappa,mu):
    T0 = pop
    S0 = height-start
    E0 = 0
    I0 = start
    A0 = start/kappa
    R0 = 0
    SI0 = start
    SA0 = 0
    y0 = [T0, S0, E0, I0, A0, R0, SI0, SA0, height]      # initial condition vector
    t  = np.linspace(0, Tmax-1, Tmax)           # time grid

    soln = odeint(tseiar, y0, t, args=(alpha,betta,sigma,gamma,thetta,height,pop,omega,kappa,mu),rtol=1e-10)
    T = soln[:, 0]
    S = soln[:, 1]
    E = soln[:, 2]
    I = soln[:, 3]
    A = soln[:, 4]
    R = soln[:, 5]
    SI= soln[:, 6]
    SA= soln[:, 7]
    N = soln[:, 8]
    
    return SI,SA


TSEAIR_,TSEAIR_2 =   getTotalInfCurveTSEIAR_g(0.0,2.0,0.025,0.07,0.000,301,100000,1,100000,0.0,0.1,0.3)
TSEAIR_M,TSEAIR_M2 = getTotalInfCurveTSEIAR_g(0.0,2.0,0.025,0.07,0.001,301,100000,1,100000,0.0,0.1,0.3)
#TSEAIR_M =  getTotalInfCurveTSEIAR_g(0.01,1.0,0.01,0.0005,0.000,301,100000-1,1,100000,0.0,0.1,0.91)

if(makegraphs):
    fig = plt.figure(figsize=(7, 4))
    plt.plot(TSEAIR_)
    #plt.plot(TSEAIR_2)
    plt.plot(TSEAIR_M)
    plt.grid(True)
    plt.xlabel("Days")
    plt.ylabel("Total cases")
    #plt.ylim(0,100000)
    #plt.legend(("Infected","Unreported infected"))
    plt.legend(("SEIUR","SEIUR modified"))
    plt.show()
    folder = "owid_graphs/"
    fig.savefig(folder+"TSEIAR_M_.eps")
    fig.savefig(folder+"TSEIAR_M_.png") 
    #print(SI.shape)
    np.savetxt("TSEIAR.txt",SI)
    np.savetxt("TSEIAR_Mod.txt",SI_M)


def GetDEFitPRS_TSI(Cases1,nNP,nNG,CPopSize,MidPoint,left,right,Name,StartArr):#########################
    D = 4    
    NP = nNP
    #NP = D*50
    NPMAX = NP
    Gap = 150
    minStart = 0
    totalI = 0
    ndays = 3
    memdays = np.zeros(ndays)
    counter = 0
    smcs1 = np.zeros(Cases1.shape[0])
    for i in range(1,Cases1.shape[0]):
        smcs1[i] = smcs1[i-1]*0.6+Cases1[i]*0.4
    for i in range(Cases1.shape[0]):
        if(i>0):
            if(smcs1[i] - smcs1[i-1] > 0.5):
                memdays[counter] = 1
            else:
                memdays[counter] = 0
            counter+= 1
            counter = counter%ndays        
        if(sum(memdays) == ndays):
            minStart = i
            break
    Pop = np.zeros((NP,D))
    Fit = np.zeros(NP)
    minStart = int(minStart)
    if(use_SI_ISO):
        for j in range(SI_iso.shape[0]):
            if(SI_iso[j] == Name):
                minStart = SI_start[j]
    #print(Cases1.shape,Gap)
    minStart = int(StartArr)
    #print("minStart",minStart)
    tempx = np.linspace(0,Cases1[minStart:].shape[0]-1,Cases1[minStart:].shape[0])    
    totalLen = Cases1.shape[0] - minStart
    TrainSize = int(MidPoint)-minStart
    #print("TrainSize",TrainSize)
    
    
    maxFES = nNG
    #maxFES = 10000*D
    FES = 0
    fitcurve = np.zeros(maxFES+NP)
    rest_ = int(0.25*maxFES)
    bestupdateFES = 0
    globalbest = 0
    globalsol = np.zeros(D)
    
    for i in range(NP):
        for j in range(D):
            if(left[j] == right[j]):
                Pop[i,j] = left[j]
            else:
                Pop[i,j] = -1
                while Pop[i,j] < left[j] or Pop[i][j] > right[j]:
                    #Pop[i,j] = np.random.normal(left[j],0.1*(right[j]-left[j]))
                    Pop[i,j] = np.random.uniform(left[j],right[j])
        tempy = getTotalInfCurveTSI(Pop[i,0],
                                    Pop[i,1],
                                    Cases1[minStart:minStart+TrainSize].shape[0],
                                    #Cases2.shape[0],
                                    Pop[i,3]*CPopSize,
                                    Cases1[minStart],
                                    #Cases2[0],
                                    CPopSize,
                                    Pop[i,2])
        Fit[i] = RE(Cases1[minStart:minStart+TrainSize],tempy)
        if(FES == 0 or Fit[i] < globalbest):
            globalbest = Fit[i]
            globalsol = Pop[i]
            bestupdateFES = FES
            #print("globalbest",globalbest,FES)
        fitcurve[FES] = globalbest
        FES = FES + 1
        
        
    Arch = np.zeros((NP,D))
    pbest = 0.20
    MemSize = 5
    Memory = np.zeros((2,MemSize))
    for i in range(MemSize):
        Memory[0,i] = 0.2
        Memory[1,i] = 0.8    
    MemoryIter = 0
    CurrentArchiveSize = 0
    
    for gen in range(maxFES):
        s = np.argsort(Fit)
        FitImpr = np.zeros(NP)
        FVals = np.zeros(NP)
        CrVals = np.zeros(NP)
        Weight = np.zeros(NP)
        NImpr = 0
        for i in range(NP):
            MemIndex = int(np.random.uniform(0,MemSize,1))
            r1 = s[int(np.random.uniform(0,NP*pbest,1))]
            r2 = int(np.random.uniform(0,NP,1))
            r3 = int(np.random.uniform(0,NP,1))     
            while(r1 == r2 or r1 == r3 or r2 == r3):
                r2 = int(np.random.uniform(0,NP,1))
                r3 = int(np.random.uniform(0,NP,1))
            xnew = np.zeros(D)
            CR = np.random.normal(Memory[1,MemIndex],0.1)
            F = -1
            while(F < 0):
                F = np.random.normal(Memory[0,MemIndex],0.1)
            if(F > 1):
                F = 1
            CR = max(min(1,CR),0)
            jrand = np.random.randint(D)
            for j in range(D):        
                if(np.random.uniform(0,1,1) < CR or jrand == j):
                    if(np.random.uniform(0,1,1) < 0.5 and CurrentArchiveSize == NP):
                        xnew[j] = Pop[i][j] + F*(Pop[r1][j] - Pop[i][j]) + F*(Pop[r2][j] - Arch[r3][j])
                    else:
                        xnew[j] = Pop[i][j] + F*(Pop[r1][j] - Pop[i][j]) + F*(Pop[r2][j] - Pop[r3][j])
                else:
                    xnew[j] = Pop[i][j] 
            for j in range(D):
                if(xnew[j] < left[j]):
                    xnew[j] = (left[j] + Pop[i][j])*0.5
                if(xnew[j] > right[j]):
                    xnew[j] = (right[j] + Pop[i][j])*0.5
            tempy = getTotalInfCurveTSI(xnew[0],
                                        xnew[1],
                                        Cases1[minStart:minStart+TrainSize].shape[0],
                                        #Cases2.shape[0],
                                        xnew[3]*CPopSize,
                                        Cases1[minStart],
                                        #Cases2[0],
                                        CPopSize,
                                        xnew[2])
            tempfit = RE(Cases1[minStart:minStart+TrainSize],tempy)            
            if(FES == 0 or Fit[i] < globalbest):
                globalbest = Fit[i]
                globalsol = Pop[i]
                bestupdateFES = FES
                #print("globalbest",globalbest,FES)
            fitcurve[FES] = globalbest
            FES = FES + 1                        
            if(tempfit < Fit[i]):            
                FitImpr[NImpr] = Fit[i]-tempfit
                FVals[NImpr] = F
                CrVals[NImpr] = CR
                Fit[i] = tempfit
                if(CurrentArchiveSize < NP):
                    Arch[CurrentArchiveSize] = Pop[i]
                    CurrentArchiveSize += 1
                else:
                    Arch[np.random.randint(NP)] = Pop[i]
                Pop[i] = np.copy(xnew)               
                NImpr += 1   
            if(FES >= maxFES):
                break
        
        if(NImpr > 0):
            SumFitDelta = np.sum(FitImpr[:NImpr])
            Weight[:NImpr] = FitImpr[:NImpr] / SumFitDelta
            sumF1 = np.sum(Weight[:NImpr]*FVals[:NImpr]*FVals[:NImpr])
            sumF2 = np.sum(Weight[:NImpr]*FVals[:NImpr])
            sumCr = np.sum(Weight[:NImpr]*CrVals[:NImpr])                 
            if(sumF2 != 0):
                Memory[0,MemoryIter] = sumF1/sumF2
            Memory[1,MemoryIter] = sumCr
        NImpr = 0
        MemoryIter += 1
        if(MemoryIter >= MemSize):
            MemoryIter = 0
                
        minfit = Fit[0]
        minindex = 0
        for i in range(NP):
            if(Fit[i] < minfit):
                minfit = Fit[i]
                minindex = i
                
        if(FES - bestupdateFES > rest_ and FES < maxFES - NP - 1):
            #print("RESTART AT",FES,bestupdateFES,rest_)            
            bestupdateFES = FES            
            for i in range(NP):
                if(Fit[i] != globalbest):
                    for j in range(D):
                        if(left[j] == right[j]):
                            Pop[i,j] = left[j]
                        else:
                            Pop[i,j] = -1
                        while Pop[i,j] < left[j] or Pop[i][j] > right[j]:
                            Pop[i,j] = np.random.uniform(left[j],right[j])
                        tempy = getTotalInfCurveTSI(Pop[i,0],
                                    Pop[i,1],
                                    Cases1[minStart:minStart+TrainSize].shape[0],
                                    #Cases2.shape[0],
                                    Pop[i,3]*CPopSize,
                                    Cases1[minStart],
                                    #Cases2[0],
                                    CPopSize,
                                    Pop[i,2])
                        Fit[i] = RE(Cases1[minStart:minStart+TrainSize],tempy)
                        fitcurve[FES] = Fit[i]
                        #print(i,Fit[i])
                        if(Fit[i] < globalbest):
                            globalbest = Fit[i]
                            globalsol = Pop[i]	
                            bestupdateFES = FES
                            #print("globalbest",globalbest,FES)
                        fitcurve[FES] = globalbest
                        FES = FES + 1
                
        
        newNP = int((4-NPMAX)/maxFES*FES+NPMAX)
        #newNP = NP
        
        #print(NP,newNP,FES,maxFES,s0)
        #print(FES,np.mean(Memory[0,:]),np.mean(Memory[1,:]))
        
        for t in range(NP-newNP):
            s0 = np.argsort(-Fit)[0]
            for i in range(s0,NP-1):
                Pop[i] = Pop[i+1]
                Fit[i] = Fit[i+1]
        NP = newNP
        
        if(FES >= maxFES):
            break

    minfit = Fit[0]
    minindex = 0
    for i in range(NP):
        if(Fit[i] < minfit):
            minfit = Fit[i]
            minindex = i
                
    return globalsol,globalbest,minStart,TrainSize,fitcurve#,Cases2
	
	
def GetDEFitPRS_TSIR(Cases1,nNP,nNG,CPopSize,MidPoint,left,right,Name,StartArr):
    #left = [0,0,0,0]
    #right = [1,0.1,0.01,1]
    #print(left,right)
    D = 5
    NP = nNP
    #NP = D*50
    NPMAX = NP
    Gap = 150
    minStart = 0
    totalI = 0
    #while(Cases1.shape[0] < Gap):
    #    Gap = np.floor(Cases1.shape[0]*0.9)      
    #totalI > np.sum(Cases1)*0.0001 and   
    ndays = 3
    memdays = np.zeros(ndays)
    counter = 0
    smcs1 = np.zeros(Cases1.shape[0])
    for i in range(1,Cases1.shape[0]):
        smcs1[i] = smcs1[i-1]*0.6+Cases1[i]*0.4
    for i in range(Cases1.shape[0]):
        #print(i)
        if(i>0):
            if(smcs1[i] - smcs1[i-1] > 0.5):
                memdays[counter] = 1
            else:
                memdays[counter] = 0
            counter+= 1
            counter = counter%ndays
        
        if(sum(memdays) == ndays):
            minStart = i
            break
        
            #totalI = Cases1[i]
            #if(totalI > 5):            
            #if(Cases1.shape[0]-minStart-10 < Gap):
            #    minStart = Cases1.shape[0]-Gap
            #break
    Pop = np.zeros((NP,D))
    Fit = np.zeros(NP)
    minStart = int(minStart)
    if(use_SI_ISO):
        for j in range(SI_iso.shape[0]):
            if(SI_iso[j] == Name):
                minStart = SI_start[j]
    #print(Cases1.shape,Gap)
    minStart = int(StartArr)
    #print("minStart",minStart)
    tempx = np.linspace(0,Cases1[minStart:].shape[0]-1,Cases1[minStart:].shape[0])    
    totalLen = Cases1.shape[0] - minStart
    #TrainSize = int(totalLen*0.75)
    TrainSize = int(MidPoint)-minStart
    #print("TrainSize",TrainSize)
    
    
    maxFES = nNG
    #maxFES = 10000*D
    FES = 0
    fitcurve = np.zeros(maxFES+NP)
    rest_ = int(0.25*maxFES)
    bestupdateFES = 0
    globalbest = 0
    globalsol = np.zeros(D)
    
    for i in range(NP):
        for j in range(D):
            if(left[j] == right[j]):
                Pop[i,j] = left[j]
            else:
                Pop[i,j] = -1
                while Pop[i,j] < left[j] or Pop[i][j] > right[j]:
                    #Pop[i,j] = np.random.normal(left[j],0.1*(right[j]-left[j]))
                    Pop[i,j] = np.random.uniform(left[j],right[j])
        tempy = getTotalInfCurveTSIR(Pop[i,0],
                                    Pop[i,1],
                                    Pop[i,2],
                                    Cases1[minStart:minStart+TrainSize].shape[0],
                                    Pop[i,4]*CPopSize,
                                    Cases1[minStart],
                                    CPopSize,
                                    Pop[i,3])
        Fit[i] = RE(Cases1[minStart:minStart+TrainSize],tempy)
        if(FES == 0 or Fit[i] < globalbest):
            globalbest = Fit[i]
            globalsol = Pop[i]
            bestupdateFES = FES
        fitcurve[FES] = globalbest
        FES = FES + 1
        
        
    Arch = np.zeros((NP,D))
    pbest = 0.20
    MemSize = 5
    Memory = np.zeros((2,MemSize))
    for i in range(MemSize):
        Memory[0,i] = 0.2
        Memory[1,i] = 0.8    
    MemoryIter = 0
    CurrentArchiveSize = 0
    
    for gen in range(maxFES):
        s = np.argsort(Fit)
        FitImpr = np.zeros(NP)
        FVals = np.zeros(NP)
        CrVals = np.zeros(NP)
        Weight = np.zeros(NP)
        NImpr = 0
        for i in range(NP):
            MemIndex = int(np.random.uniform(0,MemSize,1))
            r1 = s[int(np.random.uniform(0,NP*pbest,1))]
            r2 = int(np.random.uniform(0,NP,1))
            r3 = int(np.random.uniform(0,NP,1))     
            while(r1 == r2 or r1 == r3 or r2 == r3):
                r2 = int(np.random.uniform(0,NP,1))
                r3 = int(np.random.uniform(0,NP,1))
            xnew = np.zeros(D)
            CR = np.random.normal(Memory[1,MemIndex],0.1)
            F = -1
            while(F < 0):
                F = np.random.normal(Memory[0,MemIndex],0.1)
            if(F > 1):
                F = 1
            CR = max(min(1,CR),0)
            jrand = np.random.randint(D)
            for j in range(D):        
                if(np.random.uniform(0,1,1) < CR or jrand == j):
                    if(np.random.uniform(0,1,1) < 0.5 and CurrentArchiveSize == NP):
                        xnew[j] = Pop[i][j] + F*(Pop[r1][j] - Pop[i][j]) + F*(Pop[r2][j] - Arch[r3][j])
                    else:
                        xnew[j] = Pop[i][j] + F*(Pop[r1][j] - Pop[i][j]) + F*(Pop[r2][j] - Pop[r3][j])
                else:
                    xnew[j] = Pop[i][j] 
            for j in range(D):
                if(xnew[j] < left[j]):
                    xnew[j] = (left[j] + Pop[i][j])*0.5
                if(xnew[j] > right[j]):
                    xnew[j] = (right[j] + Pop[i][j])*0.5
            tempy = getTotalInfCurveTSIR(xnew[0],
                                        xnew[1],
                                        xnew[2],
                                        Cases1[minStart:minStart+TrainSize].shape[0],
                                        xnew[4]*CPopSize,
                                        Cases1[minStart],
                                        CPopSize,
                                        xnew[3])
            tempfit = RE(Cases1[minStart:minStart+TrainSize],tempy)            
            if(FES == 0 or Fit[i] < globalbest):
                globalbest = Fit[i]
                globalsol = Pop[i]
                bestupdateFES = FES
            fitcurve[FES] = globalbest
            FES = FES + 1            
            if(tempfit < Fit[i]):            
                FitImpr[NImpr] = Fit[i]-tempfit
                FVals[NImpr] = F
                CrVals[NImpr] = CR
                Fit[i] = tempfit
                if(CurrentArchiveSize < NP):
                    Arch[CurrentArchiveSize] = Pop[i]
                    CurrentArchiveSize += 1
                else:
                    Arch[np.random.randint(NP)] = Pop[i]
                Pop[i] = np.copy(xnew)               
                NImpr += 1   
            if(FES >= maxFES):
                break
        
        if(NImpr > 0):
            SumFitDelta = np.sum(FitImpr[:NImpr])
            Weight[:NImpr] = FitImpr[:NImpr] / SumFitDelta
            sumF1 = np.sum(Weight[:NImpr]*FVals[:NImpr]*FVals[:NImpr])
            sumF2 = np.sum(Weight[:NImpr]*FVals[:NImpr])
            sumCr = np.sum(Weight[:NImpr]*CrVals[:NImpr])                 
            if(sumF2 != 0):
                Memory[0,MemoryIter] = sumF1/sumF2
            Memory[1,MemoryIter] = sumCr
        NImpr = 0
        MemoryIter += 1
        if(MemoryIter >= MemSize):
            MemoryIter = 0
                
        minfit = Fit[0]
        minindex = 0
        for i in range(NP):
            if(Fit[i] < minfit):
                minfit = Fit[i]
                minindex = i
                
        if(FES - bestupdateFES > rest_ and FES < maxFES - NP - 1):
            bestupdateFES = FES
            for i in range(NP):
                if(Fit[i] != globalbest or True):
                    for j in range(D):
                        if(left[j] == right[j]):
                            Pop[i,j] = left[j]
                        else:
                            Pop[i,j] = -1
                        while Pop[i,j] < left[j] or Pop[i][j] > right[j]:
                            Pop[i,j] = np.random.uniform(left[j],right[j])
                        tempy = getTotalInfCurveTSIR(Pop[i,0],
                                    Pop[i,1],
                                    Pop[i,2],
                                    Cases1[minStart:minStart+TrainSize].shape[0],
                                    Pop[i,4]*CPopSize,
                                    Cases1[minStart],
                                    CPopSize,
                                    Pop[i,3])
                        Fit[i] = RE(Cases1[minStart:minStart+TrainSize],tempy)
                        fitcurve[FES] = Fit[i]
                        if(Fit[i] < globalbest):
                            globalbest = Fit[i]	
                            globalsol = Pop[i]
                            bestupdateFES = FES
                        fitcurve[FES] = globalbest
                        FES = FES + 1
                
        
        newNP = int((4-NPMAX)/maxFES*FES+NPMAX)
        s0 = np.argsort(-Fit)[0]
        #print(NP,newNP,FES,maxFES,s0)
        #print(FES,np.mean(Memory[0,:]),np.mean(Memory[1,:]))
        for t in range(NP-newNP):
            s0 = np.argsort(-Fit)[0]
            for i in range(s0,NP-1):
                Pop[i] = Pop[i+1]
                Fit[i] = Fit[i+1]
        NP = newNP
        
        if(FES >= maxFES):
            break
    
    minfit = Fit[0]
    minindex = 0
    for i in range(NP):
        if(Fit[i] < minfit):
            minfit = Fit[i]
            minindex = i
            
    return globalsol,globalbest,minStart,TrainSize,fitcurve


def GetDEFitPRS_TSEIR(Cases1,nNP,nNG,CPopSize,MidPoint,left,right,Name,StartArr):
    #left = [0,0,0,0]
    #right = [1,0.1,0.01,1]
    #print(left,right)
    D = 7
    NP = nNP
    #NP = D*50
    NPMAX = NP
    Gap = 150
    minStart = 0
    totalI = 0
    #while(Cases1.shape[0] < Gap):
    #    Gap = np.floor(Cases1.shape[0]*0.9)      
    #totalI > np.sum(Cases1)*0.0001 and   
    ndays = 3
    memdays = np.zeros(ndays)
    counter = 0
    smcs1 = np.zeros(Cases1.shape[0])
    for i in range(1,Cases1.shape[0]):
        smcs1[i] = smcs1[i-1]*0.6+Cases1[i]*0.4
    for i in range(Cases1.shape[0]):
        #print(i)
        if(i>0):
            if(smcs1[i] - smcs1[i-1] > 0.5):
                memdays[counter] = 1
            else:
                memdays[counter] = 0
            counter+= 1
            counter = counter%ndays
        
        if(sum(memdays) == ndays):
            minStart = i
            break
        
            #totalI = Cases1[i]
            #if(totalI > 5):            
            #if(Cases1.shape[0]-minStart-10 < Gap):
            #    minStart = Cases1.shape[0]-Gap
            #break
    Pop = np.zeros((NP,D))
    Fit = np.zeros(NP)
    minStart = int(minStart)
    if(use_SI_ISO):
        for j in range(SI_iso.shape[0]):
            if(SI_iso[j] == Name):
                minStart = SI_start[j]
    #print(Cases1.shape,Gap)
    minStart = int(StartArr)
    #print("minStart",minStart)
    tempx = np.linspace(0,Cases1[minStart:].shape[0]-1,Cases1[minStart:].shape[0])    
    totalLen = Cases1.shape[0] - minStart
    #TrainSize = int(totalLen*0.75)
    TrainSize = int(MidPoint)-minStart
    #print("TrainSize",TrainSize)
    
    maxFES = nNG
    #maxFES = 10000*D
    FES = 0
    fitcurve = np.zeros(maxFES+NP)
    rest_ = int(0.25*maxFES)
    bestupdateFES = 0
    globalbest = 0
    globalsol = np.zeros(D)
    
    for i in range(NP):
        for j in range(D):
            if(left[j] == right[j]):
                Pop[i,j] = left[j]
            else:
                Pop[i,j] = -1
                while Pop[i,j] < left[j] or Pop[i][j] > right[j]:
                    #Pop[i,j] = np.random.normal(left[j],0.1*(right[j]-left[j]))
                    Pop[i,j] = np.random.uniform(left[j],right[j])
        tempy = getTotalInfCurveTSEIR(Pop[i,0],
                                    Pop[i,1],
                                    Pop[i,2],
                                    Pop[i,3],
                                    Pop[i,4],
                                    Cases1[minStart:minStart+TrainSize].shape[0],
                                    Pop[i,6]*CPopSize,
                                    Cases1[minStart],
                                    CPopSize,
                                    Pop[i,5])
        Fit[i] = RE(Cases1[minStart:minStart+TrainSize],tempy)
        if(FES == 0 or Fit[i] < globalbest):
            globalbest = Fit[i]
            globalsol = Pop[i]
            bestupdateFES = FES
        fitcurve[FES] = globalbest
        FES = FES + 1
        
        
    Arch = np.zeros((NP,D))
    pbest = 0.20
    MemSize = 5
    Memory = np.zeros((2,MemSize))
    for i in range(MemSize):
        Memory[0,i] = 0.2
        Memory[1,i] = 0.8    
    MemoryIter = 0
    CurrentArchiveSize = 0
    
    for gen in range(maxFES):
        s = np.argsort(Fit)
        FitImpr = np.zeros(NP)
        FVals = np.zeros(NP)
        CrVals = np.zeros(NP)
        Weight = np.zeros(NP)
        NImpr = 0
        for i in range(NP):
            MemIndex = int(np.random.uniform(0,MemSize,1))
            r1 = s[int(np.random.uniform(0,NP*pbest,1))]
            r2 = int(np.random.uniform(0,NP,1))
            r3 = int(np.random.uniform(0,NP,1))     
            while(r1 == r2 or r1 == r3 or r2 == r3):
                r2 = int(np.random.uniform(0,NP,1))
                r3 = int(np.random.uniform(0,NP,1))
            xnew = np.zeros(D)
            CR = np.random.normal(Memory[1,MemIndex],0.1)
            F = -1
            while(F < 0):
                F = np.random.normal(Memory[0,MemIndex],0.1)
            if(F > 1):
                F = 1
            CR = max(min(1,CR),0)
            jrand = np.random.randint(D)
            for j in range(D):        
                if(np.random.uniform(0,1,1) < CR or jrand == j):
                    if(np.random.uniform(0,1,1) < 0.5 and CurrentArchiveSize == NP):
                        xnew[j] = Pop[i][j] + F*(Pop[r1][j] - Pop[i][j]) + F*(Pop[r2][j] - Arch[r3][j])
                    else:
                        xnew[j] = Pop[i][j] + F*(Pop[r1][j] - Pop[i][j]) + F*(Pop[r2][j] - Pop[r3][j])
                else:
                    xnew[j] = Pop[i][j] 
            for j in range(D):
                if(xnew[j] < left[j]):
                    xnew[j] = (left[j] + Pop[i][j])*0.5
                if(xnew[j] > right[j]):
                    xnew[j] = (right[j] + Pop[i][j])*0.5
            tempy = getTotalInfCurveTSEIR(xnew[0],
                                        xnew[1],
                                        xnew[2],
                                        xnew[3],
                                        xnew[4],
                                        Cases1[minStart:minStart+TrainSize].shape[0],
                                        xnew[6]*CPopSize,
                                        Cases1[minStart],
                                        CPopSize,
                                        xnew[5])
            tempfit = RE(Cases1[minStart:minStart+TrainSize],tempy)            
            if(FES == 0 or Fit[i] < globalbest):
                globalbest = Fit[i]
                globalsol = Pop[i]
                bestupdateFES = FES
            fitcurve[FES] = globalbest
            FES = FES + 1            
            if(tempfit < Fit[i]):            
                FitImpr[NImpr] = Fit[i]-tempfit
                FVals[NImpr] = F
                CrVals[NImpr] = CR
                Fit[i] = tempfit
                if(CurrentArchiveSize < NP):
                    Arch[CurrentArchiveSize] = Pop[i]
                    CurrentArchiveSize += 1
                else:
                    Arch[np.random.randint(NP)] = Pop[i]
                Pop[i] = np.copy(xnew)               
                NImpr += 1   
            if(FES >= maxFES):
                break
        
        if(NImpr > 0):
            SumFitDelta = np.sum(FitImpr[:NImpr])
            Weight[:NImpr] = FitImpr[:NImpr] / SumFitDelta
            sumF1 = np.sum(Weight[:NImpr]*FVals[:NImpr]*FVals[:NImpr])
            sumF2 = np.sum(Weight[:NImpr]*FVals[:NImpr])
            sumCr = np.sum(Weight[:NImpr]*CrVals[:NImpr])                 
            if(sumF2 != 0):
                Memory[0,MemoryIter] = sumF1/sumF2
            Memory[1,MemoryIter] = sumCr
        NImpr = 0
        MemoryIter += 1
        if(MemoryIter >= MemSize):
            MemoryIter = 0
                
        minfit = Fit[0]
        minindex = 0
        for i in range(NP):
            if(Fit[i] < minfit):
                minfit = Fit[i]
                minindex = i
                
        if(FES - bestupdateFES > rest_ and FES < maxFES - NP - 1):
            bestupdateFES = FES
            for i in range(NP):
                if(Fit[i] != globalbest or True):
                    for j in range(D):
                        if(left[j] == right[j]):
                            Pop[i,j] = left[j]
                        else:
                            Pop[i,j] = -1
                        while Pop[i,j] < left[j] or Pop[i][j] > right[j]:
                            Pop[i,j] = np.random.uniform(left[j],right[j])
                        tempy = getTotalInfCurveTSEIR(Pop[i,0],
                                    Pop[i,1],
                                    Pop[i,2],
                                    Pop[i,3],
                                    Pop[i,4],
                                    Cases1[minStart:minStart+TrainSize].shape[0],
                                    Pop[i,6]*CPopSize,
                                    Cases1[minStart],
                                    CPopSize,
                                    Pop[i,5])
                        Fit[i] = RE(Cases1[minStart:minStart+TrainSize],tempy)
                        fitcurve[FES] = Fit[i]
                        if(Fit[i] < globalbest):
                            globalbest = Fit[i]	
                            globalsol = Pop[i]
                            bestupdateFES = FES
                        fitcurve[FES] = globalbest
                        FES = FES + 1
                
        
        newNP = int((4-NPMAX)/maxFES*FES+NPMAX)
        s0 = np.argsort(-Fit)[0]
        #print(NP,newNP,FES,maxFES,s0)
        #print(FES,np.mean(Memory[0,:]),np.mean(Memory[1,:]))
        for t in range(NP-newNP):
            s0 = np.argsort(-Fit)[0]
            for i in range(s0,NP-1):
                Pop[i] = Pop[i+1]
                Fit[i] = Fit[i+1]
        NP = newNP
        
        if(FES >= maxFES):
            break
    
    minfit = Fit[0]
    minindex = 0
    for i in range(NP):
        if(Fit[i] < minfit):
            minfit = Fit[i]
            minindex = i
    return globalsol,globalbest,minStart,TrainSize,fitcurve


def GetDEFitPRS_TSEIAR(Cases1,nNP,nNG,CPopSize,MidPoint,left,right,Name,StartArr):
    #left = [0,0,0,0]
    #right = [1,0.1,0.01,1]
    #print(left,right)
    D = 9
    NP = nNP
    #NP = D*50 
    NPMAX = NP       
    Gap = 150
    minStart = 0
    totalI = 0
    ndays = 3
    memdays = np.zeros(ndays)
    counter = 0
    smcs1 = np.zeros(Cases1.shape[0])
    for i in range(1,Cases1.shape[0]):
        smcs1[i] = smcs1[i-1]*0.6+Cases1[i]*0.4
    for i in range(Cases1.shape[0]):
        #print(i)
        if(i>0):
            if(smcs1[i] - smcs1[i-1] > 0.5):
                memdays[counter] = 1
            else:
                memdays[counter] = 0
            counter+= 1
            counter = counter%ndays
        
        if(sum(memdays) == ndays):
            minStart = i
            break
        
            #totalI = Cases1[i]
            #if(totalI > 5):            
            #if(Cases1.shape[0]-minStart-10 < Gap):
            #    minStart = Cases1.shape[0]-Gap
            #break
    Pop = np.zeros((NP,D))
    Fit = np.zeros(NP)
    minStart = int(minStart)
    if(use_SI_ISO):
        for j in range(SI_iso.shape[0]):
            if(SI_iso[j] == Name):
                minStart = SI_start[j]
    #print(Cases1.shape,Gap)
    minStart = int(StartArr)
    #print("minStart",minStart)
    tempx = np.linspace(0,Cases1[minStart:].shape[0]-1,Cases1[minStart:].shape[0])    
    totalLen = Cases1.shape[0] - minStart
    #TrainSize = int(totalLen*0.75)
    TrainSize = int(MidPoint)-minStart
    #print("TrainSize",TrainSize)
    
    maxFES = nNG
    #maxFES = 10000*D
    FES = 0
    fitcurve = np.zeros(maxFES+NP)
    rest_ = int(0.25*maxFES)
    bestupdateFES = 0
    globalbest = 0
    globalsol = np.zeros(D)
    
    for i in range(NP):
        for j in range(D):
            if(left[j] == right[j]):
                Pop[i,j] = left[j]
            else:
                Pop[i,j] = -1
                while Pop[i,j] < left[j] or Pop[i][j] > right[j]:
                    #Pop[i,j] = np.random.normal(left[j],0.1*(right[j]-left[j]))
                    Pop[i,j] = np.random.uniform(left[j],right[j])
        tempy = getTotalInfCurveTSEIAR(Pop[i,0],
                                    Pop[i,1],
                                    Pop[i,2],
                                    Pop[i,3],
                                    Pop[i,4],
                                    Cases1[minStart:minStart+TrainSize].shape[0],
                                    Pop[i,6]*CPopSize,
                                    Cases1[minStart],
                                    CPopSize,
                                    Pop[i,5],
                                    Pop[i,7],
                                    Pop[i,8])
        Fit[i] = RE(Cases1[minStart:minStart+TrainSize],tempy)
        if(FES == 0 or Fit[i] < globalbest):
            globalbest = Fit[i]
            globalsol = Pop[i]
            bestupdateFES = FES
        fitcurve[FES] = globalbest
        FES = FES + 1
        
        
    Arch = np.zeros((NP,D))
    pbest = 0.20
    MemSize = 5
    Memory = np.zeros((2,MemSize))
    for i in range(MemSize):
        Memory[0,i] = 0.2
        Memory[1,i] = 0.8    
    MemoryIter = 0
    CurrentArchiveSize = 0
    
    for gen in range(maxFES):
        s = np.argsort(Fit)
        FitImpr = np.zeros(NP)
        FVals = np.zeros(NP)
        CrVals = np.zeros(NP)
        Weight = np.zeros(NP)
        NImpr = 0
        for i in range(NP):
            MemIndex = int(np.random.uniform(0,MemSize,1))
            r1 = s[int(np.random.uniform(0,NP*pbest,1))]
            r2 = int(np.random.uniform(0,NP,1))
            r3 = int(np.random.uniform(0,NP,1))     
            while(r1 == r2 or r1 == r3 or r2 == r3):
                r2 = int(np.random.uniform(0,NP,1))
                r3 = int(np.random.uniform(0,NP,1))
            xnew = np.zeros(D)
            CR = np.random.normal(Memory[1,MemIndex],0.1)
            F = -1
            while(F < 0):
                F = np.random.normal(Memory[0,MemIndex],0.1)
            if(F > 1):
                F = 1
            CR = max(min(1,CR),0)
            jrand = np.random.randint(D)
            for j in range(D):        
                if(np.random.uniform(0,1,1) < CR or jrand == j):
                    if(np.random.uniform(0,1,1) < 0.5 and CurrentArchiveSize == NP):
                        xnew[j] = Pop[i][j] + F*(Pop[r1][j] - Pop[i][j]) + F*(Pop[r2][j] - Arch[r3][j])
                    else:
                        xnew[j] = Pop[i][j] + F*(Pop[r1][j] - Pop[i][j]) + F*(Pop[r2][j] - Pop[r3][j])
                else:
                    xnew[j] = Pop[i][j] 
            for j in range(D):
                if(xnew[j] < left[j]):
                    xnew[j] = (left[j] + Pop[i][j])*0.5
                if(xnew[j] > right[j]):
                    xnew[j] = (right[j] + Pop[i][j])*0.5
            tempy = getTotalInfCurveTSEIAR(xnew[0],
                                        xnew[1],
                                        xnew[2],
                                        xnew[3],
                                        xnew[4],
                                        Cases1[minStart:minStart+TrainSize].shape[0],
                                        xnew[6]*CPopSize,
                                        Cases1[minStart],
                                        CPopSize,
                                        xnew[5],
                                        xnew[7],
                                        xnew[8])
            tempfit = RE(Cases1[minStart:minStart+TrainSize],tempy)            
            if(FES == 0 or Fit[i] < globalbest):
                globalbest = Fit[i]
                globalsol = Pop[i]
                bestupdateFES = FES
            fitcurve[FES] = globalbest
            FES = FES + 1            
            if(tempfit < Fit[i]):            
                FitImpr[NImpr] = Fit[i]-tempfit
                FVals[NImpr] = F
                CrVals[NImpr] = CR
                Fit[i] = tempfit
                if(CurrentArchiveSize < NP):
                    Arch[CurrentArchiveSize] = Pop[i]
                    CurrentArchiveSize += 1
                else:
                    Arch[np.random.randint(NP)] = Pop[i]
                Pop[i] = np.copy(xnew)               
                NImpr += 1   
            if(FES >= maxFES):
                break
        
        if(NImpr > 0):
            SumFitDelta = np.sum(FitImpr[:NImpr])
            Weight[:NImpr] = FitImpr[:NImpr] / SumFitDelta
            sumF1 = np.sum(Weight[:NImpr]*FVals[:NImpr]*FVals[:NImpr])
            sumF2 = np.sum(Weight[:NImpr]*FVals[:NImpr])
            sumCr = np.sum(Weight[:NImpr]*CrVals[:NImpr])                 
            if(sumF2 != 0):
                Memory[0,MemoryIter] = sumF1/sumF2
            Memory[1,MemoryIter] = sumCr
        NImpr = 0
        MemoryIter += 1
        if(MemoryIter >= MemSize):
            MemoryIter = 0
                
        minfit = Fit[0]
        minindex = 0
        for i in range(NP):
            if(Fit[i] < minfit):
                minfit = Fit[i]
                minindex = i
                
        if(FES - bestupdateFES > rest_ and FES < maxFES - NP - 1):
            bestupdateFES = FES
            for i in range(NP):
                if(Fit[i] != globalbest or True):
                    for j in range(D):
                        if(left[j] == right[j]):
                            Pop[i,j] = left[j]
                        else:
                            Pop[i,j] = -1
                        while Pop[i,j] < left[j] or Pop[i][j] > right[j]:
                            Pop[i,j] = np.random.uniform(left[j],right[j])
                        tempy = getTotalInfCurveTSEIAR(Pop[i,0],
				                    Pop[i,1],
				                    Pop[i,2],
				                    Pop[i,3],
				                    Pop[i,4],
				                    Cases1[minStart:minStart+TrainSize].shape[0],
				                    Pop[i,6]*CPopSize,
				                    Cases1[minStart],
				                    CPopSize,
				                    Pop[i,5],
				                    Pop[i,7],
				                    Pop[i,8])
                        Fit[i] = RE(Cases1[minStart:minStart+TrainSize],tempy)
                        fitcurve[FES] = Fit[i]
                        if(Fit[i] < globalbest):
                            globalbest = Fit[i]
                            globalsol = Pop[i]	
                            bestupdateFES = FES
                        fitcurve[FES] = globalbest
                        FES = FES + 1
                
        
        newNP = int((4-NPMAX)/maxFES*FES+NPMAX)
        s0 = np.argsort(-Fit)[0]
        #print(NP,newNP,FES,maxFES,s0)
        #print(FES,np.mean(Memory[0,:]),np.mean(Memory[1,:]))
        for t in range(NP-newNP):
            s0 = np.argsort(-Fit)[0]
            for i in range(s0,NP-1):
                Pop[i] = Pop[i+1]
                Fit[i] = Fit[i+1]
        NP = newNP
        
        if(FES >= maxFES):
            break
    
    minfit = Fit[0]
    minindex = 0
    for i in range(NP):
        if(Fit[i] < minfit):
            minfit = Fit[i]
            minindex = i
    return globalsol,globalbest,minStart,TrainSize,fitcurve
    
    
    
#####################################################################################################

TestNames = ["SI","BD","SIR","eSIR","SEIR","eSEIR","SEIUR","eSEIUR","eSEIURrange"]
TestType = 0
run = 0
NIndsMax = 500
NFEmax = 125000

#print(comm)

TotalNRuns = 225
RunsPerOne = 25

if(world_rank > TotalNRuns):
    pass
else:
    RunsPerProcessor = 0
    RunsPerProcessor = math.ceil(TotalNRuns/world_size)
    RunsStart = np.zeros(world_size)
    RunsEnd = np.zeros(world_size)
    for i in range(world_size):    
        RunsStart[i] = RunsPerProcessor*i
        RunsEnd[i] = RunsPerProcessor*(i+1)
        if(RunsStart[i] > TotalNRuns):
            RunsStart[i] = TotalNRuns
        if(RunsEnd[i] > TotalNRuns):
            RunsEnd[i] = TotalNRuns
        print(world_rank,RunsStart,RunsEnd)
    for tmprun in range(int(RunsStart[world_rank]),int(RunsEnd[world_rank])):
    #for tmprun in range(world_rank,world_rank+1):
        run = tmprun%RunsPerOne
        TestType = 0+int(tmprun/RunsPerOne)    
        print("TESTTYPE",TestType,TestNames[TestType])    
        #count_iter = 0

        if(TestType == 0):
            
            left = [0,0,0,0]
            right = [8,0,0,0.1]
            all_fit_err = []
            all_fit_prs = []
            for i in range(len(NamesList)):
                if(PopulationSizes[i] < 1000000):
                    all_fit_prs.append((0,0,0,0))
                    all_fit_err.append((0))
                    continue
                if(CType[i] != 0):
                    all_fit_prs.append((0,0,0,0))
                    all_fit_err.append((0))
                    continue    

                use = 1
                if(use_SI_ISO):
                    for j in range(SI_iso.shape[0]):
                        if(SI_iso[j] == NamesList[i]):
                            use = 1
                            break
                if(use == 0):
                    continue    

                Df1 = Data[Data.iloc[:,0] == NamesList[i]].total_cases
                Df1 = Df1.fillna(0)
                Cases1 = np.array(Df1)    
                Cases1 = Cases1[:int(newNCases[i])]

                curvePRS,Error,minStart,TrainSize,fitcurve = GetDEFitPRS_TSI(Cases1,NIndsMax,NFEmax,PopulationSizes[i],Cases1.shape[0]-1,left,right,NamesList[i],StartArr[i]) 
                #np.savetxt("fitcurve_"+TestNames[TestType]+"_"+str(count_iter)+".txt",fitcurve)
                #count_iter = count_iter + 1
                all_fit_prs.append(curvePRS)
                print(curvePRS,Error)
                print(Cases1[minStart:].shape[0])
                tempx = np.linspace(0,Cases1[minStart+TrainSize:].shape[0]-1,Cases1[minStart+TrainSize:].shape[0])
                tempy = getTotalInfCurveTSI(curvePRS[0],
                                            curvePRS[1],
                                            Cases1[minStart:].shape[0],
                                            curvePRS[3]*PopulationSizes[i],
                                            Cases1[minStart],
                                            PopulationSizes[i],
                                            curvePRS[2])
                tempy = tempy[TrainSize:]
                er2 = RE(Cases1[minStart+TrainSize:],tempy)      
                print(er2)
                all_fit_err.append((er2))
            FITPRS_ = np.array(all_fit_prs)
            FITERR_ = np.array(all_fit_err)
            np.savetxt("owidFITPRS_"+TestNames[TestType]+"_LSHADE_"+str(run+1)+".txt",FITPRS_)
            np.savetxt("owidFITERR_"+TestNames[TestType]+"_LSHADE_"+str(run+1)+".txt",FITERR_)

        if(TestType == 1):
            
            left = [0,0,0,0]
            right = [8,0.1,0,0.1]
            all_fit_err = []
            all_fit_prs = []
            for i in range(len(NamesList)):
                if(PopulationSizes[i] < 1000000):
                    all_fit_prs.append((0,0,0,0))
                    all_fit_err.append((0))
                    continue
                if(CType[i] != 0):
                    all_fit_prs.append((0,0,0,0))
                    all_fit_err.append((0))
                    continue    

                use = 1
                if(use_SI_ISO):
                    for j in range(SI_iso.shape[0]):
                        if(SI_iso[j] == NamesList[i]):
                            use = 1
                            break
                if(use == 0):
                    continue    

                Df1 = Data[Data.iloc[:,0] == NamesList[i]].total_cases
                Df1 = Df1.fillna(0)
                Cases1 = np.array(Df1)    
                Cases1 = Cases1[:int(newNCases[i])]

                curvePRS,Error,minStart,TrainSize,fitcurve = GetDEFitPRS_TSI(Cases1,NIndsMax,NFEmax,PopulationSizes[i],Cases1.shape[0]-1,left,right,NamesList[i],StartArr[i]) 
                all_fit_prs.append(curvePRS)
                print(curvePRS,Error)
                print(Cases1[minStart:].shape[0])
                tempx = np.linspace(0,Cases1[minStart+TrainSize:].shape[0]-1,Cases1[minStart+TrainSize:].shape[0])
                tempy = getTotalInfCurveTSI(curvePRS[0],
                                            curvePRS[1],
                                            Cases1[minStart:].shape[0],
                                            curvePRS[3]*PopulationSizes[i],
                                            Cases1[minStart],
                                            PopulationSizes[i],
                                            curvePRS[2])
                tempy = tempy[TrainSize:]
                er2 = RE(Cases1[minStart+TrainSize:],tempy)      
                print(er2)
                all_fit_err.append((er2))
            FITPRS_ = np.array(all_fit_prs)
            FITERR_ = np.array(all_fit_err)
            np.savetxt("owidFITPRS_"+TestNames[TestType]+"_LSHADE_"+str(run+1)+".txt",FITPRS_)
            np.savetxt("owidFITERR_"+TestNames[TestType]+"_LSHADE_"+str(run+1)+".txt",FITERR_)

        if(TestType == 2):
            
            left = [0,1.0/42.0,0,0,0]
            right = [8,1.0/2.0,0,0,0.1]
            all_fit_err = []
            all_fit_prs = []
            for i in range(len(NamesList)):
            #for i in range(1):
                if(PopulationSizes[i] < 1000000):
                    #print(NamesList2[i] + " less then 1m")
                    all_fit_prs.append((0,0,0,0,0))
                    all_fit_err.append((0))
                    continue
                if(CType[i] != 0):
                    #print(NamesList2[i] + " not S-curve")
                    all_fit_prs.append((0,0,0,0,0))
                    all_fit_err.append((0))
                    continue    

                use = 1
                if(use_SI_ISO):
                    for j in range(SI_iso.shape[0]):
                        if(SI_iso[j] == NamesList[i]):
                            use = 1
                            break
                if(use == 0):
                    continue    

                Df1 = Data[Data.iloc[:,0] == NamesList[i]].total_cases
                Df1 = Df1.fillna(0)
                Cases1 = np.array(Df1)    
                Cases1 = Cases1[:int(newNCases[i])]
                
                #curvePRS,Error,minStart,TrainSize = GetDEFitPRS_TSIR(Cases1,100,300,PopulationSizes[i],MidPoints[i],left,right) 
                curvePRS,Error,minStart,TrainSize,fitcurve = GetDEFitPRS_TSIR(Cases1,NIndsMax,NFEmax,PopulationSizes[i],Cases1.shape[0]-1,left,right,NamesList[i],StartArr[i]) 
                all_fit_prs.append(curvePRS)
                print(curvePRS,Error)
                print(Cases1[minStart:].shape[0])
                tempx = np.linspace(0,Cases1[minStart+TrainSize:].shape[0]-1,Cases1[minStart+TrainSize:].shape[0])
                tempy = getTotalInfCurveTSIR(curvePRS[0],
                                            curvePRS[1],
                                            curvePRS[2],
                                            Cases1[minStart:].shape[0],
                                            curvePRS[4]*PopulationSizes[i],
                                            Cases1[minStart],
                                            PopulationSizes[i],
                                            curvePRS[3]) 
                tempy = tempy[TrainSize:]
                er2 = RE(Cases1[minStart+TrainSize:],tempy)      
                #er2 = er2 / (Cases1[-1]-Cases1[minStart+TrainSize])
                print(er2)
                all_fit_err.append((er2))
                #fig = plt.figure(figsize=(10, 10))
                #plt.title(NamesList2[i]+" relative err = "+str(er2))
                #plt.plot(tempx,tempy)
                #plt.plot(tempx,Cases1[minStart+TrainSize:])   
                #plt.legend(("fitted curve","data"))
                #plt.show()             
                tempx = np.linspace(0,Cases1[minStart:].shape[0]-1,Cases1[minStart:].shape[0])
                tempy = getTotalInfCurveTSIR(curvePRS[0],
                                            curvePRS[1],
                                            curvePRS[2],
                                            Cases1[minStart:].shape[0],
                                            curvePRS[4]*PopulationSizes[i],
                                            Cases1[minStart],
                                            PopulationSizes[i],
                                            curvePRS[3]) 
                #fig = plt.figure(figsize=(10, 10))
                #plt.title(NamesList2[i]+" relative err = "+str(er2))
                #plt.plot(tempx,tempy)
                #plt.plot(tempx,Cases1[minStart:])   
                #plt.legend(("fitted curve","data"))
                #plt.show()     
            FITPRS_ = np.array(all_fit_prs)
            FITERR_ = np.array(all_fit_err)
            np.savetxt("owidFITPRS_"+TestNames[TestType]+"_LSHADE_"+str(run+1)+".txt",FITPRS_)
            np.savetxt("owidFITERR_"+TestNames[TestType]+"_LSHADE_"+str(run+1)+".txt",FITERR_)

        if(TestType == 3):
            
            left = [0,1.0/42.0,0,0,0]
            right = [8,1.0/2.0,0.1,0,0.1]
            all_fit_err = []
            all_fit_prs = []
            for i in range(len(NamesList)):
            #for i in range(1):
                if(PopulationSizes[i] < 1000000):
                    #print(NamesList2[i] + " less then 1m")
                    all_fit_prs.append((0,0,0,0,0))
                    all_fit_err.append((0))
                    continue
                if(CType[i] != 0):
                    #print(NamesList2[i] + " not S-curve")
                    all_fit_prs.append((0,0,0,0,0))
                    all_fit_err.append((0))
                    continue    

                use = 1
                if(use_SI_ISO):
                    for j in range(SI_iso.shape[0]):
                        if(SI_iso[j] == NamesList[i]):
                            use = 1
                            break
                if(use == 0):
                    continue    

                Df1 = Data[Data.iloc[:,0] == NamesList[i]].total_cases
                Df1 = Df1.fillna(0)
                Cases1 = np.array(Df1)    
                Cases1 = Cases1[:int(newNCases[i])]
                
                #curvePRS,Error,minStart,TrainSize = GetDEFitPRS_TSIR(Cases1,100,300,PopulationSizes[i],MidPoints[i],left,right) 
                curvePRS,Error,minStart,TrainSize,fitcurve = GetDEFitPRS_TSIR(Cases1,NIndsMax,NFEmax,PopulationSizes[i],Cases1.shape[0]-1,left,right,NamesList[i],StartArr[i]) 
                all_fit_prs.append(curvePRS)
                print(curvePRS,Error)
                print(Cases1[minStart:].shape[0])
                tempx = np.linspace(0,Cases1[minStart+TrainSize:].shape[0]-1,Cases1[minStart+TrainSize:].shape[0])
                tempy = getTotalInfCurveTSIR(curvePRS[0],
                                            curvePRS[1],
                                            curvePRS[2],
                                            Cases1[minStart:].shape[0],
                                            curvePRS[4]*PopulationSizes[i],
                                            Cases1[minStart],
                                            PopulationSizes[i],
                                            curvePRS[3]) 
                tempy = tempy[TrainSize:]
                er2 = RE(Cases1[minStart+TrainSize:],tempy)      
                #er2 = er2 / (Cases1[-1]-Cases1[minStart+TrainSize])
                print(er2)
                all_fit_err.append((er2))
                #fig = plt.figure(figsize=(10, 10))
                #plt.title(NamesList2[i]+" relative err = "+str(er2))
                #plt.plot(tempx,tempy)
                #plt.plot(tempx,Cases1[minStart+TrainSize:])   
                #plt.legend(("fitted curve","data"))
                #plt.show()             
                tempx = np.linspace(0,Cases1[minStart:].shape[0]-1,Cases1[minStart:].shape[0])
                tempy = getTotalInfCurveTSIR(curvePRS[0],
                                            curvePRS[1],
                                            curvePRS[2],
                                            Cases1[minStart:].shape[0],
                                            curvePRS[4]*PopulationSizes[i],
                                            Cases1[minStart],
                                            PopulationSizes[i],
                                            curvePRS[3]) 
                #fig = plt.figure(figsize=(10, 10))
                #plt.title(NamesList2[i]+" relative err = "+str(er2))
                #plt.plot(tempx,tempy)
                #plt.plot(tempx,Cases1[minStart:])   
                #plt.legend(("fitted curve","data"))
                #plt.show()     
            FITPRS_ = np.array(all_fit_prs)
            FITERR_ = np.array(all_fit_err)
            np.savetxt("owidFITPRS_"+TestNames[TestType]+"_LSHADE_"+str(run+1)+".txt",FITPRS_)
            np.savetxt("owidFITERR_"+TestNames[TestType]+"_LSHADE_"+str(run+1)+".txt",FITERR_)


        if(TestType == 4):
            
            #SEIR   alpha0 betta sigma    gamma   theta omega alpha
            left =  [0,   0,   1/42.0,   1/42.0,   0,   0,  0  ]
            right = [0.0, 8,   1/2.0,    1/2.0,    0,   0,  0.1]
            all_fit_err = []
            all_fit_prs = []
            for i in range(len(NamesList)):
            #for i in range(1):
                if(PopulationSizes[i] < 1000000):
                    #print(NamesList2[i] + " less then 1m")
                    all_fit_prs.append((0,0,0,0,0,0,0))
                    all_fit_err.append((0))
                    continue
                if(CType[i] != 0):
                    #print(NamesList2[i] + " not S-curve")
                    all_fit_prs.append((0,0,0,0,0,0,0))
                    all_fit_err.append((0))
                    continue    

                use = 1
                if(use_SI_ISO):
                    for j in range(SI_iso.shape[0]):
                        if(SI_iso[j] == NamesList[i]):
                            use = 1
                            break
                if(use == 0):
                    continue    

                Df1 = Data[Data.iloc[:,0] == NamesList[i]].total_cases
                Df1 = Df1.fillna(0)
                Cases1 = np.array(Df1)    
                Cases1 = Cases1[:int(newNCases[i])]
                #curvePRS,Error,minStart,TrainSize = GetDEFitPRS_TSEIR(Cases1,100,400,PopulationSizes[i],MidPoints[i],left,right) 
                curvePRS,Error,minStart,TrainSize,fitcurve = GetDEFitPRS_TSEIR(Cases1,NIndsMax,NFEmax,PopulationSizes[i],Cases1.shape[0]-1,left,right,NamesList[i],StartArr[i]) 
                all_fit_prs.append(curvePRS)
                print(curvePRS,Error)
                print(Cases1[minStart:].shape[0])
                tempx = np.linspace(0,Cases1[minStart+TrainSize:].shape[0]-1,Cases1[minStart+TrainSize:].shape[0])
                tempy = getTotalInfCurveTSEIR(curvePRS[0],
                                            curvePRS[1],
                                            curvePRS[2],
                                            curvePRS[3],
                                            curvePRS[4],
                                            Cases1[minStart:].shape[0],
                                            curvePRS[6]*PopulationSizes[i],
                                            Cases1[minStart],
                                            PopulationSizes[i],
                                            curvePRS[5]) 
                tempy = tempy[TrainSize:]
                er2 = RE(Cases1[minStart+TrainSize:],tempy)  
                print(minStart+TrainSize)
                #print(Cases1[-1],Cases1[minStart+TrainSize])
                #er2 = er2 / (Cases1[-1]-Cases1[minStart+TrainSize])
                print(er2)
                all_fit_err.append((er2))
                #fig = plt.figure(figsize=(10, 10))
                #plt.title(NamesList2[i]+" relative err = "+str(er2))
                #plt.plot(tempx,tempy)
                #plt.plot(tempx,Cases1[minStart+TrainSize:])   
                #plt.legend(("fitted curve","data"))
                #plt.show()             
                tempx = np.linspace(0,Cases1[minStart:].shape[0]-1,Cases1[minStart:].shape[0])
                tempy = getTotalInfCurveTSEIR(curvePRS[0],
                                            curvePRS[1],
                                            curvePRS[2],
                                            curvePRS[3],
                                            curvePRS[4],
                                            Cases1[minStart:].shape[0],
                                            curvePRS[6]*PopulationSizes[i],
                                            Cases1[minStart],
                                            PopulationSizes[i],
                                            curvePRS[5])  
                #fig = plt.figure(figsize=(10, 10))
                #plt.title(NamesList2[i]+" relative err = "+str(er2))
                #plt.plot(tempx,tempy)
                #plt.plot(tempx,Cases1[minStart:])   
                #plt.legend(("fitted curve","data"))
                #plt.show()     
            FITPRS_ = np.array(all_fit_prs)
            FITERR_ = np.array(all_fit_err)
            np.savetxt("owidFITPRS_"+TestNames[TestType]+"_LSHADE_"+str(run+1)+".txt",FITPRS_)
            np.savetxt("owidFITERR_"+TestNames[TestType]+"_LSHADE_"+str(run+1)+".txt",FITERR_)


        if(TestType == 5):
            
            #eSEIR  alpha0 betta sigma    gamma   theta omega alpha
            left =  [0,   0,   1/42.0,   1/42.0,   0,   0,  0  ]
            right = [0.0, 8,   1/2.0,    1/2.0,    0.1, 0,  0.1]
            all_fit_err = []
            all_fit_prs = []
            for i in range(len(NamesList)):
            #for i in range(1):
                if(PopulationSizes[i] < 1000000):
                    #print(NamesList2[i] + " less then 1m")
                    all_fit_prs.append((0,0,0,0,0,0,0))
                    all_fit_err.append((0))
                    continue
                if(CType[i] != 0):
                    #print(NamesList2[i] + " not S-curve")
                    all_fit_prs.append((0,0,0,0,0,0,0))
                    all_fit_err.append((0))
                    continue    

                use = 1
                if(use_SI_ISO):
                    for j in range(SI_iso.shape[0]):
                        if(SI_iso[j] == NamesList[i]):
                            use = 1
                            break
                if(use == 0):
                    continue    

                Df1 = Data[Data.iloc[:,0] == NamesList[i]].total_cases
                Df1 = Df1.fillna(0)
                Cases1 = np.array(Df1)    
                Cases1 = Cases1[:int(newNCases[i])]
                #curvePRS,Error,minStart,TrainSize = GetDEFitPRS_TSEIR(Cases1,100,400,PopulationSizes[i],MidPoints[i],left,right) 
                curvePRS,Error,minStart,TrainSize,fitcurve = GetDEFitPRS_TSEIR(Cases1,NIndsMax,NFEmax,PopulationSizes[i],Cases1.shape[0]-1,left,right,NamesList[i],StartArr[i]) 
                all_fit_prs.append(curvePRS)
                print(curvePRS,Error)
                print(Cases1[minStart:].shape[0])
                tempx = np.linspace(0,Cases1[minStart+TrainSize:].shape[0]-1,Cases1[minStart+TrainSize:].shape[0])
                tempy = getTotalInfCurveTSEIR(curvePRS[0],
                                            curvePRS[1],
                                            curvePRS[2],
                                            curvePRS[3],
                                            curvePRS[4],
                                            Cases1[minStart:].shape[0],
                                            curvePRS[6]*PopulationSizes[i],
                                            Cases1[minStart],
                                            PopulationSizes[i],
                                            curvePRS[5]) 
                tempy = tempy[TrainSize:]
                er2 = RE(Cases1[minStart+TrainSize:],tempy)  
                print(minStart+TrainSize)
                #print(Cases1[-1],Cases1[minStart+TrainSize])
                #er2 = er2 / (Cases1[-1]-Cases1[minStart+TrainSize])
                print(er2)
                all_fit_err.append((er2))
                #fig = plt.figure(figsize=(10, 10))
                #plt.title(NamesList2[i]+" relative err = "+str(er2))
                #plt.plot(tempx,tempy)
                #plt.plot(tempx,Cases1[minStart+TrainSize:])   
                #plt.legend(("fitted curve","data"))
                #plt.show()             
                tempx = np.linspace(0,Cases1[minStart:].shape[0]-1,Cases1[minStart:].shape[0])
                tempy = getTotalInfCurveTSEIR(curvePRS[0],
                                            curvePRS[1],
                                            curvePRS[2],
                                            curvePRS[3],
                                            curvePRS[4],
                                            Cases1[minStart:].shape[0],
                                            curvePRS[6]*PopulationSizes[i],
                                            Cases1[minStart],
                                            PopulationSizes[i],
                                            curvePRS[5])  
                #fig = plt.figure(figsize=(10, 10))
                #plt.title(NamesList2[i]+" relative err = "+str(er2))
                #plt.plot(tempx,tempy)
                #plt.plot(tempx,Cases1[minStart:])   
                #plt.legend(("fitted curve","data"))
                #plt.show()     
            FITPRS_ = np.array(all_fit_prs)
            FITERR_ = np.array(all_fit_err)
            np.savetxt("owidFITPRS_"+TestNames[TestType]+"_LSHADE_"+str(run+1)+".txt",FITPRS_)
            np.savetxt("owidFITERR_"+TestNames[TestType]+"_LSHADE_"+str(run+1)+".txt",FITERR_)
                
        if(TestType == 6):
               
            #SEIUR  alpha0 betta  sigma       gamma     theta omega alpha kappa    mu
            left =  [0,     0,    1.0/42.0,   1.0/42.0,   0,    0,   0,   0.0001,  0.0]
            right = [0.0,   8,    1.0/2.0,    1.0/2.0,    0,    0,   0.1, 0.5,     8.0]          
            
            all_fit_err = []
            all_fit_prs = []
            count_iter = 0
            for i in range(len(NamesList)):
            #for i in range(1):
                if(PopulationSizes[i] < 1000000):
                    print(NamesList2[i] + " less then 1m")
                    tmp = []
                    for j in range(len(left)):
                        tmp.append(0);
                    all_fit_prs.append(tmp)
                    all_fit_err.append((0))
                    continue
                if(CType[i] != 0):
                    print(NamesList2[i] + " not S-curve")
                    tmp = []
                    for j in range(len(left)):
                        tmp.append(0);
                    all_fit_prs.append(tmp)
                    all_fit_err.append((0))
                    continue

                use = 1
                if(use_SI_ISO):
                    for j in range(SI_iso.shape[0]):
                        if(SI_iso[j] == NamesList[i]):
                            use = 1
                            break
                if(use == 0):
                    continue    

                Df1 = Data[Data.iloc[:,0] == NamesList[i]].total_cases
                Df1 = Df1.fillna(0)
                Cases1 = np.array(Df1)    
                Cases1 = Cases1[:int(newNCases[i])]
                #curvePRS,Error,minStart,TrainSize = GetDEFitPRS_TSEIR(Cases1,100,400,PopulationSizes[i],MidPoints[i],left,right) 
                curvePRS,Error,minStart,TrainSize,fitcurve = GetDEFitPRS_TSEIAR(Cases1,NIndsMax,NFEmax,PopulationSizes[i],Cases1.shape[0]-1,left,right,NamesList[i],StartArr[i]) 
                #np.savetxt("fitcurve_"+TestNames[TestType]+"_"+str(count_iter)+".txt",fitcurve)
                #count_iter = count_iter + 1
                all_fit_prs.append(curvePRS)
                print(curvePRS,Error)
                print(Cases1[minStart:].shape[0])
                tempx = np.linspace(0,Cases1[minStart+TrainSize:].shape[0]-1,Cases1[minStart+TrainSize:].shape[0])
                tempy = getTotalInfCurveTSEIAR(curvePRS[0],
                                            curvePRS[1],
                                            curvePRS[2],
                                            curvePRS[3],
                                            curvePRS[4],
                                            Cases1[minStart:].shape[0],
                                            curvePRS[6]*PopulationSizes[i],
                                            Cases1[minStart],
                                            PopulationSizes[i],
                                            curvePRS[5],
                                            curvePRS[7],
                                            curvePRS[8]) 
                tempy = tempy[TrainSize:]
                er2 = RE(Cases1[minStart+TrainSize:],tempy)      
                #er2 = er2 / (Cases1[-1]-Cases1[minStart+TrainSize])
                print(er2)  
                Rt = curvePRS[7]*curvePRS[1]/curvePRS[3]+(1.0-curvePRS[7])*curvePRS[8]*curvePRS[1]/curvePRS[3]
                print("RT",Rt)
                all_fit_err.append((er2))
                #fig = plt.figure(figsize=(10, 10))
                #plt.title(NamesList2[i]+" relative err = "+str(er2))
                #plt.plot(tempx,tempy)
                #plt.plot(tempx,Cases1[minStart+TrainSize:])   
                #plt.legend(("fitted curve","data"))
                #plt.show()             
                tempx = np.linspace(0,Cases1[minStart:].shape[0]-1,Cases1[minStart:].shape[0])
                tempy = getTotalInfCurveTSEIAR(curvePRS[0],
                                            curvePRS[1],
                                            curvePRS[2],
                                            curvePRS[3],
                                            curvePRS[4],
                                            Cases1[minStart:].shape[0],
                                            curvePRS[6]*PopulationSizes[i],
                                            Cases1[minStart],
                                            PopulationSizes[i],
                                            curvePRS[5],
                                            curvePRS[7],
                                            curvePRS[8]) 
                #fig = plt.figure(figsize=(10, 10))
                #plt.title(NamesList2[i]+" relative err = "+str(er2))
                #plt.plot(tempx,tempy)
                #plt.plot(tempx,Cases1[minStart:])   
                #plt.legend(("fitted curve","data"))
                #plt.show() 

            FITPRS_ = np.array(all_fit_prs)
            FITERR_ = np.array(all_fit_err)
            np.savetxt("owidFITPRS_"+TestNames[TestType]+"_LSHADE_"+str(run+1)+".txt",FITPRS_)
            np.savetxt("owidFITERR_"+TestNames[TestType]+"_LSHADE_"+str(run+1)+".txt",FITERR_) 
                
        if(TestType == 7):
            
            #eSEIUR  alpha0 betta sigma       gamma     theta omega alpha kappa    mu
            left =  [0,     0,    1.0/42.0,   1.0/42.0,   0,    0,   0,   0.0001,  0.0]
            right = [0.0,   8,    1.0/2.0,    1.0/2.0,    0.1,  0,   0.1, 0.5,     8.0]     
            all_fit_err = []
            all_fit_prs = []
            count_iter = 0
            for i in range(len(NamesList)):
            #for i in range(1):
                if(PopulationSizes[i] < 1000000):
                    print(NamesList2[i] + " less then 1m")
                    tmp = []
                    for j in range(len(left)):
                        tmp.append(0);
                    all_fit_prs.append(tmp)
                    all_fit_err.append((0))
                    continue
                if(CType[i] != 0):
                    print(NamesList2[i] + " not S-curve")
                    tmp = []
                    for j in range(len(left)):
                        tmp.append(0);
                    all_fit_prs.append(tmp)
                    all_fit_err.append((0))
                    continue

                use = 1
                if(use_SI_ISO):
                    for j in range(SI_iso.shape[0]):
                        if(SI_iso[j] == NamesList[i]):
                            use = 1
                            break
                if(use == 0):
                    continue    

                Df1 = Data[Data.iloc[:,0] == NamesList[i]].total_cases
                Df1 = Df1.fillna(0)
                Cases1 = np.array(Df1)    
                Cases1 = Cases1[:int(newNCases[i])]
                #curvePRS,Error,minStart,TrainSize = GetDEFitPRS_TSEIR(Cases1,100,400,PopulationSizes[i],MidPoints[i],left,right) 
                curvePRS,Error,minStart,TrainSize,fitcurve = GetDEFitPRS_TSEIAR(Cases1,NIndsMax,NFEmax,PopulationSizes[i],Cases1.shape[0]-1,left,right,NamesList[i],StartArr[i])
                #np.savetxt("fitcurve_"+TestNames[TestType]+"_"+str(count_iter)+".txt",fitcurve)
                #count_iter = count_iter + 1 
                all_fit_prs.append(curvePRS)
                print(curvePRS,Error)
                print(Cases1[minStart:].shape[0])
                tempx = np.linspace(0,Cases1[minStart+TrainSize:].shape[0]-1,Cases1[minStart+TrainSize:].shape[0])
                tempy = getTotalInfCurveTSEIAR(curvePRS[0],
                                            curvePRS[1],
                                            curvePRS[2],
                                            curvePRS[3],
                                            curvePRS[4],
                                            Cases1[minStart:].shape[0],
                                            curvePRS[6]*PopulationSizes[i],
                                            Cases1[minStart],
                                            PopulationSizes[i],
                                            curvePRS[5],
                                            curvePRS[7],
                                            curvePRS[8]) 
                tempy = tempy[TrainSize:]
                er2 = RE(Cases1[minStart+TrainSize:],tempy)      
                #er2 = er2 / (Cases1[-1]-Cases1[minStart+TrainSize])
                print(er2)  
                Rt = curvePRS[7]*curvePRS[1]/curvePRS[3]+(1.0-curvePRS[7])*curvePRS[8]*curvePRS[1]/curvePRS[3]
                print("RT",Rt)
                all_fit_err.append((er2))
                #fig = plt.figure(figsize=(10, 10))
                #plt.title(NamesList2[i]+" relative err = "+str(er2))
                #plt.plot(tempx,tempy)
                #plt.plot(tempx,Cases1[minStart+TrainSize:])   
                #plt.legend(("fitted curve","data"))
                #plt.show()             
                tempx = np.linspace(0,Cases1[minStart:].shape[0]-1,Cases1[minStart:].shape[0])
                tempy = getTotalInfCurveTSEIAR(curvePRS[0],
                                            curvePRS[1],
                                            curvePRS[2],
                                            curvePRS[3],
                                            curvePRS[4],
                                            Cases1[minStart:].shape[0],
                                            curvePRS[6]*PopulationSizes[i],
                                            Cases1[minStart],
                                            PopulationSizes[i],
                                            curvePRS[5],
                                            curvePRS[7],
                                            curvePRS[8]) 
                #fig = plt.figure(figsize=(10, 10))
                #plt.title(NamesList2[i]+" relative err = "+str(er2))
                #plt.plot(tempx,tempy)
                #plt.plot(tempx,Cases1[minStart:])   
                #plt.legend(("fitted curve","data"))
                #plt.show() 

            FITPRS_ = np.array(all_fit_prs)
            FITERR_ = np.array(all_fit_err)
            np.savetxt("owidFITPRS_"+TestNames[TestType]+"_LSHADE_"+str(run+1)+".txt",FITPRS_)
            np.savetxt("owidFITERR_"+TestNames[TestType]+"_LSHADE_"+str(run+1)+".txt",FITERR_) 
                
                
        if(TestType == 8):
               
            #eSEIURr alpha0 betta sigma       gamma     theta omega alpha kappa    mu
            left =  [0,     0,    1.0/56.0,   1.0/56.0,   0,    0,   0,   0.0001,  0.0]
            right = [0.0,   12,   1.0/2.0,    1.0/2.0,    0.5,  0,   0.5, 0.5,     8.0] 

            all_fit_err = []
            all_fit_prs = []
            count_iter = 0
            for i in range(len(NamesList)):
            #for i in range(1):
                if(PopulationSizes[i] < 1000000):
                    print(NamesList2[i] + " less then 1m")
                    tmp = []
                    for j in range(len(left)):
                        tmp.append(0);
                    all_fit_prs.append(tmp)
                    all_fit_err.append((0))
                    continue
                if(CType[i] != 0):
                    print(NamesList2[i] + " not S-curve")
                    tmp = []
                    for j in range(len(left)):
                        tmp.append(0);
                    all_fit_prs.append(tmp)
                    all_fit_err.append((0))
                    continue

                use = 1
                if(use_SI_ISO):
                    for j in range(SI_iso.shape[0]):
                        if(SI_iso[j] == NamesList[i]):
                            use = 1
                            break
                if(use == 0):
                    continue    

                Df1 = Data[Data.iloc[:,0] == NamesList[i]].total_cases
                Df1 = Df1.fillna(0)
                Cases1 = np.array(Df1)    
                Cases1 = Cases1[:int(newNCases[i])]
                #curvePRS,Error,minStart,TrainSize = GetDEFitPRS_TSEIR(Cases1,100,400,PopulationSizes[i],MidPoints[i],left,right) 
                curvePRS,Error,minStart,TrainSize,fitcurve = GetDEFitPRS_TSEIAR(Cases1,NIndsMax,NFEmax,PopulationSizes[i],Cases1.shape[0]-1,left,right,NamesList[i],StartArr[i])
                #np.savetxt("fitcurve_"+TestNames[TestType]+"_"+str(count_iter)+".txt",fitcurve)
                #count_iter = count_iter + 1 
                all_fit_prs.append(curvePRS)
                print(curvePRS,Error)
                print(Cases1[minStart:].shape[0])
                tempx = np.linspace(0,Cases1[minStart+TrainSize:].shape[0]-1,Cases1[minStart+TrainSize:].shape[0])
                tempy = getTotalInfCurveTSEIAR(curvePRS[0],
                                            curvePRS[1],
                                            curvePRS[2],
                                            curvePRS[3],
                                            curvePRS[4],
                                            Cases1[minStart:].shape[0],
                                            curvePRS[6]*PopulationSizes[i],
                                            Cases1[minStart],
                                            PopulationSizes[i],
                                            curvePRS[5],
                                            curvePRS[7],
                                            curvePRS[8]) 
                tempy = tempy[TrainSize:]
                er2 = RE(Cases1[minStart+TrainSize:],tempy)      
                #er2 = er2 / (Cases1[-1]-Cases1[minStart+TrainSize])
                print(er2)  
                Rt = curvePRS[7]*curvePRS[1]/curvePRS[3]+(1.0-curvePRS[7])*curvePRS[8]*curvePRS[1]/curvePRS[3]
                print("RT",Rt)
                all_fit_err.append((er2))
                #fig = plt.figure(figsize=(10, 10))
                #plt.title(NamesList2[i]+" relative err = "+str(er2))
                #plt.plot(tempx,tempy)
                #plt.plot(tempx,Cases1[minStart+TrainSize:])   
                #plt.legend(("fitted curve","data"))
                #plt.show()             
                tempx = np.linspace(0,Cases1[minStart:].shape[0]-1,Cases1[minStart:].shape[0])
                tempy = getTotalInfCurveTSEIAR(curvePRS[0],
                                            curvePRS[1],
                                            curvePRS[2],
                                            curvePRS[3],
                                            curvePRS[4],
                                            Cases1[minStart:].shape[0],
                                            curvePRS[6]*PopulationSizes[i],
                                            Cases1[minStart],
                                            PopulationSizes[i],
                                            curvePRS[5],
                                            curvePRS[7],
                                            curvePRS[8]) 
                #fig = plt.figure(figsize=(10, 10))
                #plt.title(NamesList2[i]+" relative err = "+str(er2))
                #plt.plot(tempx,tempy)
                #plt.plot(tempx,Cases1[minStart:])   
                #plt.legend(("fitted curve","data"))
                #plt.show() 

            FITPRS_ = np.array(all_fit_prs)
            FITERR_ = np.array(all_fit_err)
            np.savetxt("owidFITPRS_"+TestNames[TestType]+"_LSHADE_"+str(run+1)+".txt",FITPRS_)
            np.savetxt("owidFITERR_"+TestNames[TestType]+"_LSHADE_"+str(run+1)+".txt",FITERR_) 
