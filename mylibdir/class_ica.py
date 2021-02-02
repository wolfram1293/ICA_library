import numpy as np
e=1e-10
class ICA:
    def __init__(self,x):
        self.x=np.matrix(x)

    def average0(self): #観測データｘの平均を0にするメソッド
        self.x-=self.x.mean(axis=1)

    def whitening(self): #白色化メソッド
        L_sigma=np.cov(self.x,bias=True)
        D, E=np.linalg.eigh(L_sigma)#固有値と固有ベクトルを求める
        D2=np.diag(np.array(D)**(-0.5))
        E=np.asmatrix(E)#ユニタリ−行列に変換
        V=E*D2*E.T# Vを導入
        z=V*self.x
        return z

    def calculate_W(self,z):# Wを求めるメソッド
        m,n=self.x.shape
        W=np.empty((0,m))
        
        for i in range(m):#データ数分wベクトルを求める
            w=np.random.rand(m,1)# wに対し初期値をランダムに選ぶ
            if w.sum()<0:# sumは正に統一
                w=w*(-1)
            w=w/np.linalg.norm(w)#正規化
            while True:
                w_prev=w
                w=np.asmatrix((np.asarray(z)*np.asarray(w.T*z)**3).mean(axis=1)).T-3*w
                w=np.linalg.qr(np.asmatrix(np.concatenate((W,w.T))).T)[0].T[-1].T#直交化法
                if w.sum()<0:# sumは正に統一
                    w=w*(-1)
                w=w/np.linalg.norm(w)#正規化
                dw=w-w_prev
                if np.linalg.norm(dw)<e:#収束していれば終了
                    W=np.concatenate((W,w.T))
                    break
                    
        return W
    
    def run(self): #独立成分分析を実行
        self.average0()#観測データｘの平均を0に
        z=self.whitening()# xを白色化
        W=self.calculate_W(z)# zからWを求める
        y=W*z # yを求める
        return y