import numpy as np
import math
import matplotlib.pyplot as plt

def r_TE(kx,ky,e1,e2):
  k=np.sqrt(kx**2+ky**2+0j)
  k2=np.sqrt(e2)
  k2x=kx
  k2y=np.sqrt(k2**2-kx**2+0j)
  #r=np.sqrt(e2/e1) * (k.real / ky.real) * (k2y.real / k2.real)
  e=np.sqrt(e2/e1)
  cosy = k2y / k2
  coso = ky / k
  r=np.sqrt(e2/e1) * cosy / coso
  r=(1-r)/(1+r)
  return (k2x,k2y,r)

def t_TE(kx,ky,e1,e2,a):
  k=np.sqrt(kx**2+ky**2+0j)
  k2=np.sqrt(e2)
  k2x=kx
  k2y=np.sqrt(k2**2-kx**2+0j)
  e=np.sqrt(e2/e1)
  cosy = k2y/ k2
  coso = ky / k
  t=e * cosy/coso
  t=2/(1+t)
  #t=t*np.exp(-a*ky.imag)
  return (k2x,k2y,t)

def r_TM(kx,ky,e1,e2):
  k=np.sqrt(kx**2+ky**2+0j)
  k2=np.sqrt(e2)
  k2x=kx
  k2y=np.sqrt(k2**2-kx**2+0j)
  e=np.sqrt(e2/e1)
  cosy = k2y / k2
  coso = ky / k
  r=(e-cosy/coso)/(e+cosy/coso)
  return (k2x,k2y,r)

def t_TM(kx,ky,e1,e2,a):
  k=np.sqrt(kx**2+ky**2+0j)
  k2=np.sqrt(e2)
  k2x=kx
  k2y=np.sqrt(k2**2-kx**2+0j)
  e=np.sqrt(e2/e1)
  cosy = k2y / k2
  coso = ky / k
  r =  cosy/coso
  r=2/(e+r)
  #r=r*np.exp(-a*ky.imag)
  return (k2x,k2y,r)

def RTE(kx,ky,e1,e2,a):

  k2=np.sqrt(e2)
  k2x=kx
  k2y=np.sqrt(k2**2-kx**2+0j)

  r1=r_TE(kx,ky,e1,e2)[2]

  r2=r_TE(k2x,k2y,e2,e1)[2]

  r3=r_TE(k2x,k2y,e2,e3)[2]

  t1=t_TE(kx,ky,e1,e2,0)[2]

  t2=t_TE(k2x,k2y,e2,e1,0)[2]

  ex=np.exp(- 2j * a * k2y)
  r=(t1*t2*r3) / (ex-r2*r3)
  r=r+r1
  return abs(r)

def RTM(kx,ky,e1,e2,a):

  k2=np.sqrt(e2)
  k2x=kx
  k2y=np.sqrt(k2**2-kx**2+0j)

  r1=r_TM(kx,ky,e1,e2)[2]

  r2=r_TM(k2x,k2y,e2,e1)[2]

  r3=r_TM(k2x,k2y,e2,e3)[2]

  t1=t_TM(kx,ky,e1,e2,0)[2]

  t2=t_TM(k2x,k2y,e2,e1,0)[2]

  ex=np.exp(- 2j * a * k2y)
  r=(t1*t2*r3) / (ex-r2*r3)
  r=r+r1
  return abs(r)

e1=1.4533**2
e2=(0.15352+4.9077j)**2
e3=1
a=0.15707*2.345


RTE(np.sin(1)*np.sqrt(e1) , np.cos(1)*np.sqrt(e1) , e1 , e2 , a)

'''
x = np.arange(0, np.pi/2, 0.00001)
plt.figure(figsize=(15, 10))
plt.plot(x, abs(r_TE(np.sin(x)*np.sqrt(e1),np.cos(x)*np.sqrt(e1),e1,e2)[2]),linewidth=2,alpha=0.8,color='black')
plt.xlabel("$Угол$ $падения$ $theta$", size = 15)
plt.ylabel("$Коэффициент$ $отражения$ $r_1$",size = 15)
plt.title(r'$r_1(theta)$ TE',size = 25)
plt.grid( True, 'major','x' )
plt.grid( True, 'major','y' )
#plt.show()

x = np.arange(0, np.pi/2, 0.00001)
plt.figure(figsize=(15, 10))
plt.plot(x, abs(r_TM(np.sin(x)*np.sqrt(e1),np.cos(x)*np.sqrt(e1),e1,e2)[2]),linewidth=2,alpha=0.8,color='black')
plt.xlabel("$Угол$ $падения$ $theta$", size = 15)
plt.ylabel("$Коэффициент$ $отражения$ $r_1$",size = 15)
plt.title(r'$r_1(theta) TM$',size = 25)
plt.grid( True, 'major','x' )
plt.grid( True, 'major','y' )
#plt.show()

x = np.arange(0, np.pi/2, 0.00001)
plt.figure(figsize=(15, 10))
plt.plot(np.pi/2-x, abs(t_TE(np.sin(x)*np.sqrt(e1),np.cos(x)*np.sqrt(e1),e1,e2,a)[2])*t_TE(t_TE(np.sin(x)*np.sqrt(e1),np.cos(x)*np.sqrt(e1),e1,e2,a)[0],t_TE(np.sin(x)*np.sqrt(e1),np.cos(x)*np.sqrt(e1),e1,e2,a)[1],e1,e2,a)[2],linewidth=2,alpha=0.8,color='black')
plt.xlabel("$Угол$ $падения$ $theta$", size = 15)
plt.ylabel("$Коэффициент$ $пропускания$ $t_3$",size = 15)
plt.title(r'$t_3(theta) TE$',size = 25)
plt.grid( True, 'major','x' )
plt.grid( True, 'major','y' )
#plt.show()

x = np.arange(0, np.pi/2, 0.00001)
plt.figure(figsize=(15, 10))
plt.plot(np.pi/2-x, abs(t_TM(np.sin(x)*np.sqrt(e1),np.cos(x)*np.sqrt(e1),e1,e2,a)[2])*t_TE(t_TE(np.sin(x)*np.sqrt(e1),np.cos(x)*np.sqrt(e1),e1,e2,a)[0],t_TE(np.sin(x)*np.sqrt(e1),np.cos(x)*np.sqrt(e1),e1,e2,a)[1],e1,e2,a)[2],linewidth=2,alpha=0.8,color='black')
plt.xlabel("$Угол$ $падения$ $theta$", size = 15)
plt.ylabel("$Коэффициент$ $пропускания$ $t_3$",size = 15)
plt.title(r'$t_3(theta) TM$',size = 25)
plt.grid( True, 'major','x' )
plt.grid( True, 'major','y' )
#plt.show()

x = np.arange(0, np.pi/2, 0.000001)
plt.figure(figsize=(15, 10))
plt.plot(x, RTE(np.sin(x)*np.sqrt(e1) , np.cos(x)*np.sqrt(e1) , e1 , e2 , a) ,linewidth=2,alpha=0.8,color='black')
plt.xlabel("$Угол$ $падения$ $theta$", size = 15)
plt.ylabel("$Амплитуда$ $отражения$ $R$",size = 15)
plt.title(r'$R(theta) TE$',size = 25)
plt.grid( True, 'major','x' )
plt.grid( True, 'major','y' )
#plt.show()
'''
x = np.arange(0, np.pi/2, 0.000001)
plt.figure(figsize=(15, 10))
plt.plot(x, RTM( np.sin(x) * np.sqrt(e1) , np.cos(x) * np.sqrt(e1) , e1 , e2 , a),linewidth=2,alpha=0.8,color='black')
plt.xlabel("$Угол$ $падения$ $theta$", size = 15)
plt.ylabel("$Амплитуда$ $отражения$ $R$",size = 15)
plt.title(r'$R(theta) TM$',size = 25)
plt.grid( True, 'major','x' )
plt.grid( True, 'major','y' )
plt.show()
'''
x = np.arange(0, np.pi/2, 0.000001)
plt.figure(figsize=(15, 10))
plt.plot(x, RTE( np.sin( x ) * np.sqrt( e1 ) , np.cos( x ) * np.sqrt( e1 ) , e1 , e2 , a),linewidth=2,alpha=0.8,color='black')
plt.plot(x, RTM(np.sin(x)*np.sqrt(e1),np.cos(x)*np.sqrt(e1),e1,e2,a),linewidth=2,alpha=0.8,color='red')
plt.xlabel("$Угол$ $падения$ $theta$", size = 15)
plt.ylabel("$Амплитуда$ $отражения$ $R$",size = 15)
plt.title(r'$R(theta)$',size = 25)
plt.grid( True, 'major','x' )
plt.grid( True, 'major','y' )
plt.show()
'''

from mpl_toolkits.mplot3d import Axes3D
'''
#ax = axes3d.Axes3D(plt.figure(dpi=1000))

X = np.arange(0, np.pi/2, 0.00001)
Y = np.arange(0.0001, 100, 0.00001)



#z=RTE(np.sin(x[i])*np.sqrt(e1),np.sin(x[i])*np.sqrt(e1),e1 ,e2 , y[i])

#X, Y, Z = np.meshgrid(x, y, z)
fig = plt.figure()
#ax = plt.axes(projection='3d')
axes3d.plot_surface(X, Y, RTE(np.sin(X)*np.sqrt(e1),np.cos(X)*np.sqrt(e1),e1 ,e2 , Y))
#ax.contour3D(X, Y, Z, 50, cmap='binary')
plt.xlabel('$Ось X$')
plt.ylabel('$Ось Y$')
ax.set_zlabel('Z axis')
ax.view_init(50, 40)
plt.show()


u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = u
y = v
#z = np.cos(v)
z = RTE( np.sin(u) * np.sqrt(e1) , np.cos(u)*np.sqrt(e1) , e1 ,e2 , v)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='inferno')
#ax.legend()
plt.show()


fig, ax = plt.subplots(nrows=1, ncols=1, num=0, figsize=(16, 8),
                       subplot_kw={'projection': '3d'})
gridY, gridX = np.mgrid[-4:4:33 * 1j, -4:4:33 * 1j]
#Z = np.sin(np.sqrt(gridX ** 2 + gridY ** 2))
Z = RTE(gridX, np.sin(gridX) * np.sqrt(e1) , np.cos(gridX) * np.sqrt(e1) , e1 ,e2 , gridY)
pSurf = ax.plot_trisurf(gridX, gridY, Z, rstride=1, cstride=1, cmap='rainbow')
fig.colorbar(pSurf)
plt.show()


n_radii = 8
n_angles = 36
radii = np.linspace(0.125, 1.0, n_radii)
angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)

angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)

x = np.append(0, (radii).flatten())
y = np.append(0, (angles).flatten())
z = RTE(x, np.sin(x) * np.sqrt(e1) , np.cos(x) * np.sqrt(e1) , e1 ,e2 , y)

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)

plt.show()
'''
