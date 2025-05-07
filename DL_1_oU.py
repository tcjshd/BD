def fnet(x):
  if x>=0:
    return 1
  else:
    return 0


w1=1
w2=0
b=-2
c=1
x1=[0,1,0,1]
x2=[0,0,1,1]
d=[0,1,1,1]
epochs=10
for epoch in range(epochs):
  print(f"Epoch {epoch}")
  error_count = 0
  for i in range(4):
    net = x1[i]*w1 + x2[i]*w2 + b
    O = fnet(net)
    e = d[i] - O
    delta_w1 = c*e*x1[i]
    delta_w2 = c*e*x2[i]
    delta_b = e
    w1+=delta_w1
    w2+=delta_w2
    b+=delta_b
    print(f"x1 = {x1[i]:2} | x2 = {x2[i]:2} | d = {d[i]:2} | net = {net:3} | y = {O:2} | e = {e:2} | " f"Δw1 = {delta_w1:2} | Δw2 = {delta_w2:2} | Δb = {delta_b:2} | w1 = {w1:2} | w2 = {w2:2} | b = {b:2}")
    if e != 0:
      error_count += 1

  if error_count == 0:
    print(f"Convergence met in Epoch {epoch}")
    break