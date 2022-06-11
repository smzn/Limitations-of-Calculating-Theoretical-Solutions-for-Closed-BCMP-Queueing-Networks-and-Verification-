import numpy as np
from numpy.linalg import solve
import pandas as pd
import time
import sys
import csv
import math
import datetime
from mpi4py import MPI
import itertools
import psutil
import random

class BCMP_MVA_Computation:
    
    def __init__(self, N, R, K, mu, type_list, m, K_total, rank, size, comm):
        self.N = N #網内の拠点数(プログラム内部で指定する場合は1少なくしている)
        self.R = R #クラス数
        self.K = K #網内の客数 K = [K1, K2]のようなリスト。トータルはsum(K)
        self.K_total = K_total
        self.rank = rank
        self.size = size
        self.comm = comm
        #Transition Probability generate
        p = [[]]
        if rank == 0:
            p = self.getTransitionProbability(self.N, self.R)
        self.p = comm.bcast(p, root=0)	
        #print(self.p)
        #print(self.p.shape)
        self.alpha = self.getArrival(self.p)
        self.mu = mu #サービス率 (N×R)
        self.type_list = type_list #Type1(FCFS),Type2プロセッサシェアリング（Processor Sharing: PS）,Type3無限サーバ（Infinite Server: IS）,Type4後着順割込継続型（LCFS-PR)
        self.m = m #今回は窓口数1(複数窓口は実装できていない)
        print('rank = {0}'.format(self.rank))
        
        #並列計算で全ての計算過程を保持しないように変更する(2022/01/23)
        self.km = (np.max(self.K)+1)**self.R #[2,1]なので3進法で考える(3^2=9状態) #[0,0]->0, [0,1]->1, [1,0]->3, [1,1]->4, [2,0]->6, [2,1]->7
        #self.L = np.zeros((self.N, self.R, self.km),dtype = 'float32') #平均形内人数 (self.Lを作成するとR=5で作成できなくなる 2022/02/04)
        #self.T = np.zeros((self.N, self.R, self.km),dtype = 'float32') #平均系内時間
        #self.lmd = np.zeros((self.R, self.km),dtype = 'float32') #各クラスのスループット
        self.start = time.time()
        #self.process_text = './process/process08.txt'
        self.process_text = './process/process_N'+str(self.N)+'_R'+str(self.R)+'_K'+str(self.K_total)+'_Core'+str(self.size)+'.txt'
        self.cpu_list = []
        self.mem_list = []
        self.roop_time = []
        self.combi_len = []

    def getMVA(self):
        state_list = [] #ひとつ前の階層の状態番号
        l_value_list =[] #ひとつ前の状態に対するLの値
        state_dict = {} #ひとつ前の状態に対する{l_value_list, state_list}の辞書
        last_L = []
        for k_index in range(1, self.K_total+1):
            if self.rank == 0:
                with open(self.process_text, 'a') as f:
                    mem = psutil.virtual_memory()
                    cpu = psutil.cpu_percent(interval=1)
                    print('k = {0}, Memory = {1}GB, CPU = {2}, elapse = {3}'.format(k_index, mem.used/10**9, cpu, time.time() - self.start), file=f)
                    self.cpu_list.append(cpu)
                    self.mem_list.append(mem.used/10**9)
                    self.roop_time.append(time.time() - self.start)
            '''
            print('rank = {0}, k_index = {1}'.format(self.rank, k_index))
            if self.rank == 0:
                with open('./process/process.txt', 'a') as f:
                    print('組み合わせ作成', file=f)
            '''
            k_combi_list_div_all = [[]for i in range(self.size)]
            if self.rank == 0: #rank0だけが組み合わせを作成
                k_combi_list = self.getCombiList4(self.K, k_index)#20220316 getCombiList5 -> getCombiList4
                self.combi_len.append(len(k_combi_list))
                with open(self.process_text, 'a') as f:
                    print('Combination = {0}'.format(len(k_combi_list)), file=f)
                    print(k_combi_list, file=f)
                #with open(self.process_text, 'a') as f:
                #    print('k_combi_list作成', file=f)
                #k_combi_listをsize分だけ分割
                size_index = 0
                for k_combi_list_val in k_combi_list:
                    k_combi_list_div_all[size_index % self.size].append(k_combi_list_val)
                    size_index += 1
                for rank in range(1, self.size):
                    self.comm.send(k_combi_list_div_all[rank], dest=rank, tag=20)
                k_combi_list_div = k_combi_list_div_all[0]
                #with open(self.process_text, 'a') as f:
                #    print('k_combi_list_div送付', file=f)
            #self.comm.barrier() #プロセス同期
            else :
                k_combi_list_div = self.comm.recv(source=0, tag=20)         
            
            #if self.rank == 0:
            #    with open(self.process_text, 'a') as f:    
            #        print('len(k_combi_list) = {0}, 総組み合わせ算出, elapse = {1}'.format(len(k_combi_list), time.time() - self.start), file=f)
            
            '''
            #k_combi_listをsize分だけ分割
            k_combi_list_div = [[] for i in range(self.size)]
            size_index = 0
            for k_combi_list_val in k_combi_list:
                k_combi_list_div[size_index % self.size].append(k_combi_list_val)
                size_index += 1
           '''
            #with open(self.process_text, 'a') as f:    
            #    print('rank = {2}, len(k_combi_list_div) = {0}, 組み合わせ分割, elapse = {1}'.format(len(k_combi_list_div), time.time() - self.start, self.rank), file=f)
            #print('k_combi_list_all = {0}'.format(k_combi_list_div))
            #print('rank = {0}, k_combi_list_div = {1}'.format(self.rank, k_combi_list_div[self.rank]))
            #for idx, val in enumerate(self.combi_list):
            #print('k_combi_list_div length = {0}'.format(len(k_combi_list_div[self.rank])))
            
            
            #並列ループ内での受け渡しに利用
            L = np.zeros((self.N, self.R, len(k_combi_list_div)),dtype = 'float32') #平均形内人数 
            T = np.zeros((self.N, self.R, len(k_combi_list_div)),dtype = 'float32') #平均系内時間
            lmd = np.zeros((self.R, len(k_combi_list_div)),dtype = 'float32') #各クラスのスループット

            for idx, val in enumerate(k_combi_list_div):#自分の担当だけ実施
                #Tの更新
                #k_state = self.getState(val) #kの状態を10進数に変換
                #print('Index : {0}, k = {1}, state = {2}'.format(idx, val, k_state))

                for n in range(self.N): #P336 (8.43)
                    for r in range(self.R):
                        if self.type_list[n] == 3:
                            #self.T[n,r, k_state] = 1 / self.mu[r,n]
                            T[n,r, idx] = 1 / self.mu[r,n]
                        else:
                            r1 = np.zeros(self.R) #K-1rを計算
                            r1[r] = 1 #対象クラスのみ1
                            k1v = val - r1 #ベクトルの引き算
                            #print('n = {0}, r = {1}, k1v = {2}, k = {3}'.format(n,r,k1v, val))

                            if np.min(k1v) < 0: #k-r1で負になる要素がある場合
                                continue

                            sum_l = 0
                            for i in range(self.R):#k-1rを状態に変換
                                if np.min(k1v) >= 0: #全ての状態が0以上のとき(一応チェック)
                                    #getState_time = time.time()
                                    kn = self.getState(k1v)
                                    #if self.rank == 0: #getStateは時間はかからない
                                    #    with open(self.process_text, 'a') as f:
                                    #        print('self.getState(k1v)での処理時間 : {0}'.format(time.time() - getState_time), file=f)
                                    #sum_l += self.L[n, i, int(kn)] #knを整数型に変換 (self.Lを利用しない 2022/02/04)
                                    #state_list_idx = self.list_index(state_list[n*self.R:(n+1)*self.R], kn) #前回の情報で更新 state_listは現在のn,rの場合で持ってくる (2022/02/04)
                                    list_index_time = time.time()
                                    #state_list_idx = self.list_index(state_list, kn)
                                    l_value = state_dict.get((kn,n,i))#state_listで検索して、l_valueを返す
                                    #if self.rank == 0:
                                    #    with open(self.process_text, 'a') as f:
                                    #        print('self.list_index(state_list, kn)での処理時間 : {0}'.format(time.time() - list_index_time), file=f)
                                    if l_value is not None:
                                        sum_l += l_value 
                                    #print(state_list[n*self.R:(n+1)*self.R])
                                    #print('今回の状態番号 : {0}'.format(kn))
                                    #print(state_list)
                                    #print('state_list_idx : {0}'.format(state_list_idx))
                                    #print(l_value_list)
                                    #if state_list_idx >=0:
                                        #print(l_value_list[n*self.R:(n+1)*self.R][state_list_idx])
                                        #sum_l += l_value_list[n*self.R:(n+1)*self.R][state_list_idx] 
                                        #print('今回のl_value_list[{1}]の値 : {0}'.format(l_value_list[state_list_idx + n * self.R + i],state_list_idx + n * self.R + i))
                                        #sum_l += l_value_list[state_list_idx + n * self.R + r] 
                            if self.m[n] == 1: #P336 (8.43) Type-1,2,4 (m_i=1)
                                #print('n = {0}, r = {1}, k_state = {2}'.format(n,r,k_state))
                                #self.T[n, r, k_state] = 1 / self.mu[r, n] * (1 + sum_l)
                                T[n, r, idx] = 1 / self.mu[r, n] * (1 + sum_l)
                #print('T = {0}'.format(self.T))
                #if self.rank == 0:
                #    with open(self.process_text, 'a') as f:    
                #        print('k = {0}, idx = {1}, T計算完了, elapse = {2}'.format(k_index, idx, time.time() - self.start), file=f)

                #λの更新
                for r in range(self.R):
                    sum = 0
                    for n in range(self.N):
                        #sum += self.alpha[r,n] * self.T[n,r,k_state]
                        sum += self.alpha[r,n] * T[n,r,idx]
                    if sum > 0:
                        #self.lmd[r,k_state] = val[r] / sum
                        lmd[r,idx] = val[r] / sum
                    #print('r = {0}, k = {1},lambda = {2}'.format(r, val, self.lmd[r,k_state]))
                #if self.rank == 0:
                #    with open(self.process_text, 'a') as f:    
                #        print('k = {0}, idx = {1}, λ計算完了, elapse = {2}'.format(k_index, idx, time.time() - self.start), file=f)

                #Gの更新
                ''' #rの扱いをどうしたらいい？(要確認)
                r1 = np.zeros(R) #K-1rを計算
                r1[r] = 1 #対象クラスのみ1
                k1v = val - r1 #ベクトルの引き算
                kn = getState(K,R,k1v)
                print('kn = {0}'.format(kn))
                print('lamda = {0}'.format())
                G[k_state] = G[int(kn)] / lmd[r,int(kn)]
                '''

                #Lの更新
                for n in range(self.N):#P336 (8.47)
                    for r in range(self.R):
                        #self.L[n,r,k_state] = self.lmd[r,k_state] * self.T[n,r,k_state] * self.alpha[r,n]
                        L[n,r,idx] = lmd[r,idx] * T[n,r,idx] * self.alpha[r,n]
                        #print('n = {0}, r = {1}, k = {2}, L = {3}'.format(n,r,val,L[n,r,k_state]))

                ''' self.Lは利用しない (2022/02/04)
                #aggregation to self.T,lmd,L at self.rank == 0
                if self.rank == 0:
                    for idx, j in enumerate(k_combi_list_div[self.rank]):
                        k_state = self.getState(j) 
                        #for n in range(self.N): #Update T (Tとλはまとめる必要がない)
                        #    for r in range(self.R):
                        #        self.T[n, r, k_state] = T[n, r, idx]
                        #for r in range(self.R): #Update Lambda
                        #    self.lmd[r,k_state] = lmd[r,idx]
                        for n in range(self.N):#Update L
                            for r in range(self.R):
                                self.L[n,r,k_state] = L[n,r,idx]
            
                '''   
                #if self.rank == 0:
                #    with open(self.process_text, 'a') as f:    
                #        print('k = {0}, idx = {1}, L計算完了, elapse = {2}'.format(k_index, idx, time.time() - self.start), file=f)
                        
                        
            #全体の処理を集約してからブロードキャスト
            state_list = []
            l_value_list =[]
            state_dict = {} #辞書利用(2022/02/11)
            n_list = [] #20220320 add
            r_list = [] #20220320 add
            if self.rank == 0:
                #for idx, j in enumerate(k_combi_list_div[0]): #rank == 0の情報をまとめる
                for idx, j in enumerate(k_combi_list_div): #rank == 0の情報をまとめる
                    k_state = self.getState(j)
                    for n in range(self.N):#Lの更新
                        for r in range(self.R):
                            state_list.append(k_state) #self.Lの代わりにこれを渡す(2022/02/03)
                            l_value_list.append(L[n,r,idx]) #self.Lの代わりにこれを渡す(2022/02/03)
                            n_list.append(n) #20220320
                            r_list.append(r) #20220320
                for i in range(1, self.size):
                    #k_combi_list_div_rank = self.comm.recv(source=i, tag=11)
                    #T_rank = self.comm.recv(source=i, tag=12)
                    #lmd_rank = self.comm.recv(source=i, tag=13)
                    l_rank = self.comm.recv(source=i, tag=14) #Lのみ集約
                    #print('receive : {0}, {1}'.format(i, l_rank))
                    
                    #リストの結合
                    for idx, j in enumerate(k_combi_list_div_all[i]):
                        #k_state = self.getState(k_combi_list_div_rank[j]) #kの状態を10進数に変換
                        k_state = self.getState(j) #kの状態を10進数に変換
                        #for n in range(self.N): #Tの更新
                        #    for r in range(self.R):
                        #        self.T[n, r, k_state] = T_rank[n, r, idx]
                        #for r in range(self.R): #Lambdaの更新
                        #    self.lmd[r,k_state] = lmd_rank[r,idx]
                        for n in range(self.N):#Lの更新
                            for r in range(self.R):
                                #self.L[n,r,k_state] = l_rank[n,r,idx] #利用しない(2022/02/04)
                                state_list.append(k_state) #self.Lの代わりにこれを渡す(2022/02/03)
                                l_value_list.append(l_rank[n,r,idx]) #self.Lの代わりにこれを渡す(2022/02/03)
                                n_list.append(n) #20220320
                                r_list.append(r) #20220320
                #self.comm.barrier() #プロセス同期
                #print(self.T)
                #print(self.lmd)
                #print(self.L)
            
            else:
                #self.comm.send(k_combi_list_div[self.rank], dest=0, tag=11)
                #self.comm.send(T, dest=0, tag=12)
                #self.comm.send(lmd, dest=0, tag=13)
                self.comm.send(L, dest=0, tag=14)
            self.comm.barrier() #プロセス同期
            
            if self.rank == 0:
                with open(self.process_text, 'a') as f:
                    print('k = {0}, 集約完了, elapse = {1}'.format(k_index, time.time() - self.start), file=f)
            
            #ここでブロードキャストする
            #self.T = self.comm.bcast(self.T, root=0)
            #self.lmd = self.comm.bcast(self.lmd, root=0)
            #self.L = self.comm.bcast(self.L, root=0) #self.Lをブロードキャストするとエラーになるのでやめる(2022/02/03)
            state_list = self.comm.bcast(state_list, root=0)
            l_value_list = self.comm.bcast(l_value_list, root=0)
            if k_index == self.K_total:
                last_L = l_value_list
            n_list = self.comm.bcast(n_list, root=0)
            r_list = self.comm.bcast(r_list, root=0)
            # 辞書に直す(2022/02/11)
            #state_dict = dict(zip(l_value_list,state_list))#state_listで検索して、l_value_listを返す
            state_dict = dict(zip(zip(state_list, n_list, r_list),l_value_list)) #20220320
            ''' self.Lを利用しない 2022/02/04
            if self.rank != 0: #各プロセスでself.Lに集約(2022/02/03)
                for n in range(self.N):
                    for r in range(self.R):
                        for i,j in zip(state_list, l_value_list): #このfor文はいらない？単に一つずつindexを増やして入れればいい
                            self.L[n,r,i] = j
             '''
            
            if self.rank == 0:
                with open(self.process_text, 'a') as f:
                    print('k = {0}, ブロードキャスト完了, elapse = {1}'.format(k_index, time.time() - self.start), file=f)
            
        #平均系内人数最終結果
        #last = self.getState(self.combi_list[-1]) #combi_listの最終値が最終結果の人数
        #last = self.getState(self.K) #combi_listの最終値が最終結果の人数 -> 利用しない(2022/02/04)
        #L_index = {'class0': self.L[:,0,last], 'class1' : self.L[:,1,last]} #クラス2個の場合
        #L_index = {'class0': self.L[:,0,last]} #クラス1つの場合
        #df_L = pd.DataFrame(L_index)
        #df_L.to_csv('/content/drive/MyDrive/研究/BCMP/csv/MVA_L(N:'+str(self.N)+',R:'+str(self.R)+',K:'+str(self.K)+').csv')
        #return self.L[:,:,last]
        
        if self.rank == 0:
            df_info = pd.DataFrame({ 'combination': self.combi_len, 'memory' : self.mem_list, 'cpu' : self.cpu_list, 'elapse' : self.roop_time})
            df_info.to_csv('./tp/camputation_info_N'+str(N)+'_R'+str(R)+'_K'+str(self.K_total)+'_Core'+str(self.size)+'.csv', index=True)
		
        return last_L
            
    def list_index(self, l, val, default=-1):#リスト内に値がある場合、その要素番号を返す、ない場合は-1を返す(2022/02/04)
        if val in l:
            return l.index(val)
        else:
            return default    
        
    def getState(self, k):#k=[k1,k2,...]を引数としたときにn進数を返す(R = len(K))
        k_state = 0
        for i in range(self.R): #Lを求めるときの、kの状態を求める(この例では3進数)
            k_state += k[i]*((np.max(self.K)+1)**(self.R-1-i))
        return k_state

    def getArrival(self, p):#マルチクラスの推移確率からクラス毎の到着率を計算する
        p = np.array(p) #リストからnumpy配列に変換(やりやすいので)
        alpha = np.zeros((self.R, self.N))
        for r in range(self.R): #マルチクラス毎取り出して到着率を求める
            alpha[r] = self.getCloseTraffic(p[r * self.N : (r + 1) * self.N, r * self.N : (r + 1) * self.N])
        return alpha
    
    #閉鎖型ネットワークのノード到着率αを求める
    #https://mizunolab.sist.ac.jp/2019/09/bcmp-network1.html
    def getCloseTraffic(self, p):
        e = np.eye(len(p)-1) #次元を1つ小さくする
        pe = p[1:len(p), 1:len(p)].T - e #行と列を指定して次元を小さくする
        lmd = p[0, 1:len(p)] #0行1列からの値を右辺に用いる
        try:
            slv = solve(pe, lmd * (-1)) #2021/09/28 ここで逆行列がないとエラーが出る
        except np.linalg.LinAlgError as err: #2021/09/29 Singular Matrixが出た時は、対角成分に小さい値を足すことで対応 https://www.kuroshum.com/entry/2018/12/28/python%E3%81%A7singular_matrix%E3%81%8C%E8%B5%B7%E3%81%8D%E3%82%8B%E7%90%86%E7%94%B1
            print('Singular Matrix')
            pe += e * 0.00001 
            slv = solve(pe, lmd * (-1)) 
        #lmd *= -1
        #slv = np.linalg.pinv(pe) * lmd #疑似逆行列で求める
        alpha = np.insert(slv, 0, 1.0) #α1=1を追加
        return alpha    

    def getCombiList2(self, combi, K, R, idx, Klist):
        if len(combi) == R:
            #print(combi)
            Klist.append(combi.copy())
            #print(Klist)
            return Klist
        for v in range(K[idx]+1):
            combi.append(v)
            Klist = self.getCombiList2(combi, K, R, idx+1, Klist)
            combi.pop()
        return Klist
    
    
    def getCombiList4(self, K, Pnum): #並列計算用：Pnumを増やしながら並列計算(2022/1/19)
        #Klist各拠点最大人数 Pnum足し合わせ人数
        Klist = [[j for j in range(K[i]+1)] for i in range(len(K))]
        combKlist = list(itertools.product(*Klist))
        combK = [list(cK) for cK in combKlist if sum(cK) == Pnum ]
        return combK
    
    
    def getCombiList5(self, K, Pnum): #(2022/02/08)
        #Klist各拠点最大人数 Pnum足し合わせ人数
        #最大数の指定
        num = Pnum
        if Pnum > np.max(K):
            num = np.max(K)
        #print(np.max(K))
        Klist = [j for j in range(num + 1)] #利用する数値の指定
        combK = [list(cK) for cK in list(itertools.combinations_with_replacement(Klist, len(K))) if sum(cK) == Pnum ]
        #print(len(combK))
        #print(sys.getsizeof(combK))
        return combK

    def getTransitionProbability(self):
        pr = np.zeros((self.R*self.N, self.R*self.N))
        for r in range(self.R):
            class_number = 0
            while class_number != 1:
                p = np.random.rand(self.N, self.N)
                for i, val in enumerate(np.sum(p, axis=1)):
                    p[i] /= val
                for i in range(self.N):
                    for j in range(self.N):
                        pr[r*self.N+i,r*self.N+j] = p[i,j]
                equivalence, class_number = self.getEquivalence(0, 5, p)
                if class_number == 1:
                    break
        return pr

    def getEquivalence(self, th, roop, p):
        list_number = 0 

        #1.
        equivalence = [[] for i in range(len(p))] 
        
        #2.
        for ix in range(roop):
            p = np.linalg.matrix_power(p.copy(), ix+1) 
            for i in range(len(p)):
                for j in range(i+1, len(p)):
                    if(p[i][j] > th and p[j][i] > th):
                        #3. 
                        find = 0 
                        for k in range(len(p)):
                            if i in equivalence[k]:
                                find = 1 
                                if j not in equivalence[k]:
                                    equivalence[k].append(j)        
                                break
                            if j in equivalence[k]: 
                                find = 1 
                                if i not in equivalence[k]:
                                    equivalence[k].append(i)        
                                break
                        if(find == 0):
                            equivalence[list_number].append(i)
                            if(i != j):
                                equivalence[list_number].append(j)
                            list_number += 1

        #4.
        for i in range(len(p)):
            find = 0
            for j in range(len(p)):
                if i in equivalence[j]:
                    find = 1
                    break
            if find == 0:
                equivalence[list_number].append(i)
                list_number += 1

        #5.
        class_number = 0
        for i in range(len(p)):
            if len(equivalence[i]) > 0:
                class_number += 1

        return equivalence, class_number

	#重力モデルで推移確率行列を作成 
    def getGravity(self, distance, popularity): #distanceは距離行列、popularityはクラス分の人気度
        C = 0.1475
        alpha = 1.0
        beta = 1.0
        eta = 0.5
        class_number = len(popularity[0]) #クラス数
        tp = np.zeros((len(distance) * class_number, len(distance) * class_number))
        for r in range(class_number):
            for i in range(len(distance) * r, len(distance) * (r+1)):
                for j in range(len(distance) * r, len(distance) * (r+1)):
                    if distance[i % len(distance)][j % len(distance)] > 0:
                        tp[i][j] = C * (popularity[i % len(distance)][r]**alpha) * (popularity[j % len(distance)][r]**beta) / (distance[i % len(distance)][j % len(distance)]**eta)
        row_sum = np.sum(tp, axis=1) #行和を算出
        for i in range(len(tp)): #行和を1にする
            if row_sum[i] > 0:
                tp[i] /= row_sum[i]
        return tp

    def getTransitionProbability(self, N, R): #20220313追加
	#(1) 拠点の設置と拠点間距離
        node_position_x = [random.randint(0,500) for i in range(N)]
        node_position_y = [random.randint(0,500) for i in range(N)]
        from_id = [] #DF作成用
        to_id = [] #DF作成用
        distance = []
        for i in range(N):
            for j in range(i+1,N):
                from_id.append(i)
                to_id.append(j)
                distance.append(np.sqrt((node_position_x[i]-node_position_x[j])**2 + (node_position_y[i]-node_position_y[j])**2))
        df_distance = pd.DataFrame({ 'from_id' : from_id, 'to_id' : to_id, 'distance' : distance })#データフレーム化
	#距離行列の作成
        distance_matrix = np.zeros((N,N))
        for row in df_distance.itertuples(): #右三角行列で作成される
            distance_matrix[int(row.from_id)][int(row.to_id)] = row.distance
        for i in range(len(distance_matrix)): #下三角に値を入れる(対象)
            for j in range(i+1, len(distance_matrix)):
                distance_matrix[j][i] = distance_matrix[i][j]
		
	#(2)人気度の設定
        popularity = np.abs(np.random.normal(10, 2, (N, R)))
		
	#(3)推移確率行列の作成
        tp = self.getGravity(distance_matrix, popularity)
		
	#(4)拠点情報(拠点番号、位置(x,y)、人気度(クラス数分))の生成
        df_node = pd.DataFrame({ 'node_number' : range(N), 'position_x' : node_position_x, 'position_y' : node_position_y})
        df_node.set_index('node_number',inplace=True)
	#popularityを追加
        columns = ['popurarity_'+str(i) for i in range(R)]
        for i, val in enumerate(columns):
            df_node[val] = popularity[:, i]
		
	#(5)情報の保存
        df_node.to_csv('./tp/node_N'+str(N)+'_R'+str(R)+'_K'+str(self.K_total)+'_Core'+str(self.size)+'.csv', index=True)
        df_distance.to_csv('./tp/distance_N'+str(N)+'_R'+str(R)+'_K'+str(self.K_total)+'_Core'+str(self.size)+'.csv', index=True)
        pd.DataFrame(distance_matrix).to_csv('./tp/distance_matrix_N'+str(N)+'_R'+str(R)+'_K'+str(self.K_total)+'_Core'+str(self.size)+'.csv', index=True)
        pd.DataFrame(tp).to_csv('./tp/transition_probability_N'+str(N)+'_R'+str(R)+'_K'+str(self.K_total)+'_Core'+str(self.size)+'.csv', index=True)
		
        return tp


if __name__ == '__main__':
   
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    #推移確率行列に合わせる
    N = int(sys.argv[1])
    R = int(sys.argv[2])
    K_total = int(sys.argv[3]) 
    #N = 33 #33
    #R = 2
    #K_total = 500
    K = [(K_total + i) // R for i in range(R)] #クラス人数を自動的に配分する
    mu = np.full((R, N), 1) #サービス率を同じ値で生成(サービス率は調整が必要)
    type_list = np.full(N, 1) #サービスタイプはFCFS
    m = np.full(N, 1) #今回は窓口数1(複数窓口は実装できていない)
    #p = pd.read_csv('/content/drive/MyDrive/研究/BCMP/csv/transition33.csv')
    #bcmp = BCMP_MVA(N, R, K, mu, type_list, p, m)
    bcmp = BCMP_MVA_Computation(N, R, K, mu, type_list, m, K_total, rank, size, comm)
    start = time.time()
    L = bcmp.getMVA()
    elapsed_time = time.time() - start
    print ("calclation_time:{0}".format(elapsed_time) + "[sec]")
    if rank == 0:
        print('L = \n{0}'.format(L))
        Lr = np.zeros((N, R))
        for n in range(N):
            for r in range(R):
                print('L[{0},{1},{3}] = {2}'.format(n, r, L[(n*R+r)],n*R+r))
                Lr[n, r] = L[(n*R+r)]
        pd.DataFrame(Lr).to_csv('./tp/L_N'+str(N)+'_R'+str(R)+'_K'+str(K_total)+'_Core'+str(size)+'.csv', index=True)
    #mpiexec -n 8 python3 BCMP_MVA_Computation_v9.py 33 2 100