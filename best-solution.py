# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

#!pip install gurobipy
import gurobipy as gp
from gurobipy import Model, GRB, quicksum

# %% [markdown]
# # **<font color="#FBBF44">Functions</font>**

# %%
# Time limits for different sets of instances
timelimCHU = 1200
timelimCE = 3600
timelimG = 7200

# Cost factors for large and Small vehicles
a1 = np.random.randint(5,10)/10 # = .8
a2 = a1*.75 # = .6

# Instance generator
def instance_gen(seed, clients, S1, S2, Q1, dmin, dmax):
    np.random.seed(seed) # Initial seed
    n = clients # number of clients
    # random location of clients within 8km radius
    xc = np.random.rand(n+1)*8 
    yc = np.random.rand(n+1)*8
    points = range(0,n+1)

    N = [i for i in range(1, n + 1)] # Set of clients
    V = [0] + N # Set of all nodes, including depot
    A = [(i,j) for i in V for j in V if i != j] # Available arcs between nodes
    
    S = S1 + S2 # Total amount of available vehicles
    K = [i for i in range(1, S + 1)] # Number of available vehicles, including large and small
    K1 = [i for i in range(1, S1 + 1)] # subset of large vehicles
    K2 = [i for i in range(S1+1, S1+S2+1)] # subset of small vehicles
    
    Ak = [(i,j,k) for k in K for i in V for j in V if i != j] # Available arcs between nodes for each vehicle
    Q = {} # Capacities
    for i in K:
        if i in K1:
            Q[i] = Q1 
        else:
            Q[i] = Q1/2

    d = {i: np.random.randint(dmin, dmax) for i in N} # Random demands from customers

    a1 = .8
    a2 = .6

    d1 = {(i,j,1): (np.hypot(xc[i] - xc[j], yc[i] - yc[j])) + np.random.rand() for i, j in A}
    d2 = {(i,j,2): d1[i,j,1] for i, j in A}
    c1 = {k:v*a1 for k, v in d1.items()} # Cost for traversing i,j with LargeVehicle
    c2 = {k:v*a2 for k, v in d2.items()} # Cost for traversing i,j with SmallVehicle
    c = dict(c1)
    c.update(c2) 
    return n, xc, yc, points, list(N), V, K, K1, K2, A, Ak, S, Q, d, c

# %%
# Building the model
def build_model():
    model = gp.Model(name)
    x = model.addVars(A, K, vtype=GRB.BINARY) # x[i,j,k] = equals to 1 if vehicle k traverses i,j, 0 otherwise
    y = model.addVars(N, K, vtype=GRB.INTEGER) # y[i,k] = amount delivered to client i by vehicle k 
    u = model.addVars(N, K, vtype=GRB.INTEGER) # u[i,k] = support variable for MTZ constraints
    z = model.addVars(N, K, vtype=GRB.BINARY) # z[i,k] = equals to 1 if client i is served by vehicle k, 0 otherwise
    M = 1000000000 # Big M

    # objective
    model.setObjective(quicksum(c[i,j,1]*x[i,j,k] for i, j in A if i != j for k in K1) + quicksum(c[i,j,2]*x[i,j,k] for i, j in A if i != j for k in K2), GRB.MINIMIZE)
    # (1)
    model.addConstrs(quicksum(x[i,h,k] for i in V if i != h) - quicksum(x[h,j,k] for j in V if j != h) == 0 for h in V for k in K)
    # (2)
    model.addConstrs(quicksum(x[0,j,k] for j in V if j!=0) == 1 for k in K)
    # (3)
    model.addConstrs(quicksum(y[i,k] for k in K) == d[i] for i in N)
    # (4)
    model.addConstrs(quicksum(y[i,k] for i in N) <= Q[k] for k in K)
    # (5)
    model.addConstrs(d[i]*z[i,k] >= y[i,k] for i in N for k in K)
    # (6)
    model.addConstrs(z[i,k] == quicksum(x[i,j,k] for j in V if j!=i) for i in N for k in K)
    # (7)
    model.addConstrs(u[i,k] + 1 <= u[j,k] + M*(1 - x[i,j,k]) for i, j in A if i != 0 and j != 0 for k in K)
    # (8)
    model.addConstrs(u[i,k] <= Q[k] for i in N for k in K)
    model.addConstrs(u[i,k] >= d[i]*x[i,j,k] for i, j in A if i != 0 and j != 0 for k in K)
    
    return model, x

# %%
# Plotting solution
def plot_sol(xc, yc, K1, K2, active_arcs, name):
    colors = []
    for i in range(len(active_arcs)):
        colors.append('#%06X' % np.random.randint(0, 0xFFFFFF))

    plt.plot(xc[0], yc[0], c='red', marker='s', label="Depot")
    plt.scatter(xc[1:], yc[1:], label="Clients")
#     for i in range(len(xc)):
#         plt.annotate(points[i], (xc[i] - .5, yc[i] + 0.4))
    
    for k in K1:
        for i,j,k in active_arcs:
            plt.annotate(text='', xy=(xc[j],yc[j]), xytext=(xc[i],yc[i]), zorder=0, 
                         arrowprops=dict(arrowstyle= '->, head_length=.5', color = colors[k], lw=1.5, mutation_scale=15))
    for k in K2:
        for i,j,k in active_arcs:
            plt.annotate(text='', xy=(xc[j],yc[j]), xytext=(xc[i],yc[i]), zorder=0, 
                         arrowprops=dict(arrowstyle= '->, head_length=.5', color = colors[k], lw=1.5, mutation_scale=15))

    plt.xlim((-0.5, 8.5))
    plt.ylim((-0.5, 8.5))
    handles, labels = plt.gca().get_legend_handles_labels()
    lines = {}
    K = K1 + K2
    for k in K:
        if k in K1:
            lines[k] = Line2D([0], [0], label='LargeVehicle {}'.format(k), color = colors[k]) #, color='teal')
        else:
            lines[k] = Line2D([0], [0], label='SmallVehicle {}'.format(k), color = colors[k]) #, color='teal')

    handles.extend(lines.values())

    plt.legend(handles=handles, bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.5)
    plt.title("Optimal routes for {}".format(name))
    #plt.savefig("FigXX-Tour{}".format(name)+".png", bbox_inches='tight', dpi=600) 
    plt.show()

# %%
# Finding the edges from solution values, as a tuplelist for each k
def selected(vals):
    s = {k:gp.tuplelist() for k in K}
    for i, j, k in vals.keys():
        if vals[i,j,k] > 0.99:
            s[k].append((i,j))
    return s
       
# Given the edges, finding the optimal route for each k
def subtour(edges):
    nodes = set(i for e in edges for i in e)
    unvisited = list(nodes)
    cycle = list(nodes)
    while unvisited:  
        thiscycle = []
        neighbors = unvisited
        while neighbors:
            current = neighbors[0]
            thiscycle.append(current)
            unvisited.remove(current)
            neighbors = [j for i, j in edges.select(current, '*')
                         if j in unvisited]
        if len(thiscycle) <= len(cycle): # even if it's the same, we reuse it so that we get the final tour in order
            cycle = thiscycle # New shortest subtour
    return cycle

# Print each tour
def print_route(objective, edges):
    print(f"The optimal cost for the distance traveled is: {(round(objective,2))} €")
    for k in K:
        tour = subtour(edges[k])
        tour.append(0) # return to depot
        print ("Route for vehicle k%i: %s" % (k, " -> ".join(map(str, tour))))

# %% [markdown]
# # **<font color="#BBBF">CHU INSTANCES</font>**

# %%
times = []
objectives = []

# %% [markdown]
# ## **<font color="#FBBF44">CHU-01</font>**

# %%
############ Instance Initialization CHU-01 ############
name = "CHU-01"
n, xc, yc, points, N, V, K, K1, K2, A, Ak, S, Q, d, c = instance_gen(seed=5,
                                                                     clients=5,
                                                                     S1=1,
                                                                     S2=1,
                                                                     Q1=30,
                                                                     dmin=7,
                                                                     dmax=10)
############ model1 Construction ############
model1, x = build_model()

############ model1 Solving ############
model1.reset()
model1.Params.TimeLimit = timelimCHU  # Time limit
model1.optimize()

############ Plotting Solution ############
if model1.solcount >= 1:
    #model1.write(name+".lp")
    active_arcs = [a for a in Ak if x[a].x > .99]
    plot_sol(xc, yc, K1, K2, active_arcs, name)

    vals = model1.getAttr('X', x)
    print_route(objective=model1.objVal, edges=selected(vals))
    print(f"Time to best solution = {model1.runtime}")
    times.append(model1.runtime)
    objectives.append(model1.objVal)
else:
    print("No feasible solution found")

# %% [markdown]
# ## **<font color="#FBBF44">CHU-02</font>**

# %%
############ Instance Initialization CHU-02 ############
name = "CHU-02"
n, xc, yc, points, N, V, K, K1, K2, A, Ak, S, Q, d, c = instance_gen(seed=4,
                                                                     clients=10,
                                                                     S1=1,
                                                                     S2=1,
                                                                     Q1=70,
                                                                     dmin=7,
                                                                     dmax=10)
############ model2 Construction ############
model2, x = build_model()

############ model2 Solving ############
model2.reset()
model2.Params.TimeLimit = timelimCHU  # Time limit
model2.optimize()

############ Plotting Solution ############
if model2.solcount >= 1:
    # model2.write(name+".lp")
    active_arcs = [a for a in Ak if x[a].x > .99]
    plot_sol(xc, yc, K1, K2, active_arcs, name)

    vals = model2.getAttr('X', x)
    print_route(objective=model2.objVal, edges=selected(vals))
    times.append(model2.runtime)
    print(f"Time to best solution = {model2.runtime}")
    objectives.append(model2.objVal)
else:
    print("No feasible solution found")

# %% [markdown]
# ## **<font color="#FBBF44">CHU-03</font>**

# %%
############ Instance Initialization CHU-03 ############
name = "CHU-03"
n, xc, yc, points, N, V, K, K1, K2, A, Ak, S, Q, d, c = instance_gen(seed=3,
                                                                     clients=15,
                                                                     S1=1,
                                                                     S2=2,
                                                                     Q1=80,
                                                                     dmin=7,
                                                                     dmax=10)
############ model3 Construction ############
model3, x = build_model()

############ model3 Solving ############
model3.reset()
model3.Params.TimeLimit = timelimCHU  # Time limit
model3.optimize()

############ Plotting Solution ############
if model3.solcount >= 1:
    # model3.write(name+".lp")
    active_arcs = [a for a in Ak if x[a].x > .99]
    plot_sol(xc, yc, K1, K2, active_arcs, name)

    vals = model3.getAttr('X', x)
    print_route(objective=model3.objVal, edges=selected(vals))
    times.append(model3.runtime)
    print(f"Time to best solution = {model3.runtime}")
    objectives.append(model3.objVal)
else:
    print("No feasible solution found")

# %% [markdown]
# ## **<font color="#FBBF44">CHU-04</font>**

# %%
############ Instance Initialization CHU-04 ############
name = "CHU-04"
n, xc, yc, points, N, V, K, K1, K2, A, Ak, S, Q, d, c = instance_gen(seed=2,
                                                                     clients=20,
                                                                     S1=2,
                                                                     S2=2,
                                                                     Q1=70,
                                                                     dmin=7,
                                                                     dmax=10)
############ model4 Construction ############
model4, x = build_model()

############ model4 Solving ############
model4.reset()
model4.Params.TimeLimit = timelimCHU  # Time limit
model4.Params.MIPGap = .15
model4.optimize()

############ Plotting Solution ############
if model4.solcount >= 1:
    # model4.write(name+".lp")
    active_arcs = [a for a in Ak if x[a].x > .99]
    plot_sol(xc, yc, K1, K2, active_arcs, name)

    vals = model4.getAttr('X', x)
    print_route(objective=model4.objVal, edges=selected(vals))
    times.append(model4.runtime)
    print(f"Time to best solution = {model4.runtime}")
    objectives.append(model4.objVal)
else:
    print("No feasible solution found")

# %% [markdown]
# ## **<font color="#FBBF44">CHU-05</font>**

# %%
############ Instance Initialization CHU-05 ############
name = "CHU-05"
n, xc, yc, points, N, V, K, K1, K2, A, Ak, S, Q, d, c = instance_gen(seed=1,
                                                                     clients=25,
                                                                     S1=2,
                                                                     S2=3,
                                                                     Q1=75,
                                                                     dmin=7,
                                                                     dmax=10)
############ model5 Construction ############
model5, x = build_model()

############ model5 Solving ############
model5.reset()
model5.Params.TimeLimit = timelimCHU  # Time limit
model5.Params.MIPGap = .05
model5.optimize()

############ Plotting Solution ############
if model5.solcount >= 1:
    # model5.write(name+".lp")
    active_arcs = [a for a in Ak if x[a].x > .99]
    plot_sol(xc, yc, K1, K2, active_arcs, name)

    vals = model5.getAttr('X', x)
    print_route(objective=model5.objVal, edges=selected(vals))
    times.append(model5.runtime)
    print(f"Time to best solution = {model5.runtime}")
    objectives.append(model5.objVal)
else:
    print("No feasible solution found")

# %% [markdown]
# # **<font color="#BBBF">CE INSTANCES</font>**

# %% [markdown]
# ## **<font color="#FBBF44">CE-01</font>**

# %%
############ Instance Initialization CE-01 ############
name = "CE-01"
n, xc, yc, points, N, V, K, K1, K2, A, Ak, S, Q, d, c = instance_gen(seed=1,
                                                                     clients=40,
                                                                     S1=1,
                                                                     S2=1,
                                                                     Q1=400,
                                                                     dmin=12,
                                                                     dmax=15)
############ model6 Construction ############
model6, x = build_model()

############ model6 Solving ############
model6.reset()
model6.Params.TimeLimit = timelimCE  # Time limit
model6.Params.MIPGap = .01
model6.optimize()

############ Plotting Solution ############
if model6.solcount >= 1:
    # model6.write(name+".lp")
    active_arcs = [a for a in Ak if x[a].x > .99]
    plot_sol(xc, yc, K1, K2, active_arcs, name)

    vals = model6.getAttr('X', x)
    print_route(objective=model6.objVal, edges=selected(vals))
    times.append(model6.runtime)
    print(f"Time to best solution = {model6.runtime}")
    objectives.append(model6.objVal)
else:
    print("No feasible solution found")

# %% [markdown]
# ## **<font color="#FBBF44">CE-02</font>**

# %%
############ Instance Initialization CE-02 ############
name = "CE-02"
n, xc, yc, points, N, V, K, K1, K2, A, Ak, S, Q, d, c = instance_gen(seed=2,
                                                                     clients=50,
                                                                     S1=3,
                                                                     S2=3,
                                                                     Q1=170,
                                                                     dmin=12,
                                                                     dmax=15)
############ model7 Construction ############
model7, x = build_model()

############ model7 Solving ############
model7.reset()
model7.Params.TimeLimit = timelimCE  # Time limit
model7.Params.MIPGap = .15
model7.optimize()

############ Plotting Solution ############
if model7.solcount >= 1:
    # model7.write(name+".lp")
    active_arcs = [a for a in Ak if x[a].x > .99]
    plot_sol(xc, yc, K1, K2, active_arcs, name)

    vals = model7.getAttr('X', x)
    print_route(objective=model7.objVal, edges=selected(vals))
    times.append(model7.runtime)
    print(f"Time to best solution = {model7.runtime}")
    objectives.append(model7.objVal)
else:
    print("No feasible solution found")

# %% [markdown]
# ## **<font color="#FBBF44">CE-03</font>**

# %%
############ Instance Initialization CE-03 ############
name = "CE-03"
n, xc, yc, points, N, V, K, K1, K2, A, Ak, S, Q, d, c = instance_gen(seed=3,
                                                                     clients=60,
                                                                     S1=5,
                                                                     S2=3,
                                                                     Q1=140,
                                                                     dmin=12,
                                                                     dmax=15)
############ model8 Construction ############
model8, x = build_model()

############ model8 Solving ############
model8.reset()
model8.Params.TimeLimit = timelimCE  # Time limit
model8.Params.MIPGap = .15
model8.optimize()

############ Plotting Solution ############
if model8.solcount >= 1:
    # model8.write(name+".lp")
    active_arcs = [a for a in Ak if x[a].x > .99]
    plot_sol(xc, yc, K1, K2, active_arcs, name)

    vals = model8.getAttr('X', x)
    print_route(objective=model8.objVal, edges=selected(vals))
    times.append(model8.runtime)
    print(f"Time to best solution = {model8.runtime}")
    objectives.append(model8.objVal)
else:
    print("No feasible solution found")

# %% [markdown]
# ## **<font color="#FBBF44">CE-04</font>**

# %%
############ Instance Initialization CE-04 ############
name = "CE-04"
n, xc, yc, points, N, V, K, K1, K2, A, Ak, S, Q, d, c = instance_gen(seed=4,
                                                                     clients=75,
                                                                     S1=3,
                                                                     S2=5,
                                                                     Q1=210,
                                                                     dmin=12,
                                                                     dmax=15)
############ model9 Construction ############
model9, x = build_model()

############ model9 Solving ############
model9.reset()
model9.Params.TimeLimit = timelimCE  # Time limit
model9.optimize()

############ Plotting Solution ############
if model9.solcount >= 1:
    # model9.write(name+".lp")
    active_arcs = [a for a in Ak if x[a].x > .99]
    plot_sol(xc, yc, K1, K2, active_arcs, name)

    vals = model9.getAttr('X', x)
    print_route(objective=model9.objVal, edges=selected(vals))
    times.append(model9.runtime)
    print(f"Time to best solution = {model9.runtime}")
    objectives.append(model9.objVal)
else:
    print("No feasible solution found")

# %% [markdown]
# ## **<font color="#FBBF44">CE-05</font>**

# %%
############ Instance Initialization CE-05 ############
name = "CE-05"
n, xc, yc, points, N, V, K, K1, K2, A, Ak, S, Q, d, c = instance_gen(seed=5,
                                                                     clients=100,
                                                                     S1=5,
                                                                     S2=3,
                                                                     Q1=250,
                                                                     dmin=12,
                                                                     dmax=15)
############ model10 Construction ############
model10, x = build_model()

############ model10 Solving ############
model10.reset()
model10.Params.TimeLimit = timelimCE # Time limit
model10.Params.MIPGap = .25
model10.optimize()

############ Plotting Solution ############
if model10.solcount >= 1:
    # model10.write(name+".lp")
    active_arcs = [a for a in Ak if x[a].x > .99]
    plot_sol(xc, yc, K1, K2, active_arcs, name)

    vals = model10.getAttr('X', x)
    print_route(objective=model10.objVal, edges=selected(vals))
    times.append(model10.runtime)
    print(f"Time to best solution = {model10.runtime}")
    objectives.append(model10.objVal)
else:
    print("No feasible solution found")

# %% [markdown]
# # **<font color="#BBBF">G INSTANCES</font>**

# %% [markdown]
# ## **<font color="#FBBF44">G-01</font>**

# %%
############ Instance Initialization G-01 ############
name = "G-01"
n, xc, yc, points, N, V, K, K1, K2, A, Ak, S, Q, d, c = instance_gen(seed=1,
                                                                     clients=120,
                                                                     S1=1,
                                                                     S2=1,
                                                                     Q1=800,
                                                                     dmin=5,
                                                                     dmax=10)
############ model11 Construction ############
model11, x = build_model()

############ model11 Solving ############
model11.reset()
model11.Params.TimeLimit = timelimG # Time limit
model5.Params.MIPGap = .20
model11.optimize()

############ Plotting Solution ############
if model11.solcount >= 1:
    # model11.write(name+".lp")
    active_arcs = [a for a in Ak if x[a].x > .99]
    plot_sol(xc, yc, K1, K2, active_arcs, name)

    vals = model11.getAttr('X', x)
    print_route(objective=model11.objVal, edges=selected(vals))
    times.append(model11.runtime)
    print(f"Time to best solution = {model11.runtime}")
    objectives.append(model11.objVal)
else:
    print("No feasible solution found")

# %% [markdown]
# ## **<font color="#FBBF44">G-02</font>**

# %%
############ Instance Initialization G-02 ############
name = "G-02"
n, xc, yc, points, N, V, K, K1, K2, A, Ak, S, Q, d, c = instance_gen(seed=2,
                                                                     clients=140,
                                                                     S1=2,
                                                                     S2=1,
                                                                     Q1=560,
                                                                     dmin=5,
                                                                     dmax=10)
############ model12 Construction ############
model12, x = build_model()

############ model12 Solving ############
model12.reset()
model12.Params.TimeLimit = timelimG # Time limit
model12.Params.MIPGap = .20
model12.optimize()

############ Plotting Solution ############
if model12.solcount >= 1:
    # model12.write(name+".lp")
    active_arcs = [a for a in Ak if x[a].x > .99]
    plot_sol(xc, yc, K1, K2, active_arcs, name)

    vals = model12.getAttr('X', x)
    print_route(objective=model12.objVal, edges=selected(vals))
    times.append(model12.runtime)
    print(f"Time to best solution = {model12.runtime}")
    objectives.append(model12.objVal)
else:
    print("No feasible solution found")

# %% [markdown]
# ## **<font color="#FBBF44">G-03</font>**

# %%
############ Instance Initialization G-03 ############
name = "G-03"
n, xc, yc, points, N, V, K, K1, K2, A, Ak, S, Q, d, c = instance_gen(seed=3,
                                                                     clients=160,
                                                                     S1=2,
                                                                     S2=2,
                                                                     Q1=650,
                                                                     dmin=5,
                                                                     dmax=10)
############ model13 Construction ############
model13, x = build_model()

############ model13 Solving ############
model13.reset()
model13.Params.TimeLimit = timelimG  # Time limit
model13.Params.MIPGap = .30
model13.optimize()

############ Plotting Solution ############
if model13.solcount >= 1:
    # model13.write(name+".lp")
    active_arcs = [a for a in Ak if x[a].x > .99]
    plot_sol(xc, yc, K1, K2, active_arcs, name)

    vals = model13.getAttr('X', x)
    print_route(objective=model13.objVal, edges=selected(vals))
    times.append(model13.runtime)
    print(f"Time to best solution = {model13.runtime}")
    objectives.append(model13.objVal)
else:
    print("No feasible solution found")

# %% [markdown]
# ## **<font color="#FBBF44">G-04</font>**

# %%
############ Instance Initialization G-04 ############
name = "G-04"
n, xc, yc, points, N, V, K, K1, K2, A, Ak, S, Q, d, c = instance_gen(seed=4,
                                                                     clients=180,
                                                                     S1=3,
                                                                     S2=1,
                                                                     Q1=600,
                                                                     dmin=5,
                                                                     dmax=10)
############ model14 Construction ############
model14, x = build_model()

############ model14 Solving ############
model14.reset()
model14.Params.TimeLimit = timelimG # Time limit
model14.Params.MIPGap = .30
model14.optimize()

############ Plotting Solution ############
if model14.solcount >= 1:
    # model14.write(name+".lp")
    active_arcs = [a for a in Ak if x[a].x > .99]
    plot_sol(xc, yc, K1, K2, active_arcs, name)

    vals = model14.getAttr('X', x)
    print_route(objective=model14.objVal, edges=selected(vals))
    times.append(model14.runtime)
    print(f"Time to best solution = {model14.runtime}")
    objectives.append(model14.objVal)
else:
    print("No feasible solution found")

# %% [markdown]
# ## **<font color="#FBBF44">G-05</font>**

# %%
############ Instance Initialization G-05 ############
name = "G-05"
n, xc, yc, points, N, V, K, K1, K2, A, Ak, S, Q, d, c = instance_gen(seed=5,
                                                                     clients=200,
                                                                     S1=3,
                                                                     S2=2,
                                                                     Q1=585,
                                                                     dmin=5,
                                                                     dmax=10)
############ model15 Construction ############
model15, x = build_model()

############ model15 Solving ############
model15.reset()
model15.Params.TimeLimit = timelimG  # Time limit
model15.Params.MIPGap = .30
model15.optimize()

############ Plotting Solution ############
if model15.solcount >= 1:
    # model15.write(name+".lp")
    active_arcs = [a for a in Ak if x[a].x > .99]
    plot_sol(xc, yc, K1, K2, active_arcs, name)

    vals = model15.getAttr('X', x)
    print_route(objective=model15.objVal, edges=selected(vals))
    times.append(model15.runtime)
    print(f"Time to best solution = {model15.runtime}")
    objectives.append(model15.objVal)
else:
    print("No feasible solution found")

# %% [markdown]
# # **<font color="#BBBF">Plotting results</font>**

# %%
# Runtime results
times_1stSolution = [
    0.0129525146484375,
    0.01496124267578125,
    0.06981086730957031,
    0.04587745666503906,
    0.08788490295410156,
    0.7049407958984375,
    0.396087646484375,
    1.4575767517089844,
    1.27703857421875,
    10.33984375,
    23.35746955871582,
    143.83235931396484,
    1234.250093460083,
    2782.9001121520996, 
    4954.151718139648]

times_bestSolution = [
    0.04388618469238281,
    0.3703422546386719,
    8.246297836303711,
    680.0794353485107,
    1093.8382225036621, 
    771.6032619476318,
    2736.157049179077,
    3183.7309646606445,
    3442.1439895629883,
    3564.2493629455566,
    6627.279033660889,
    6517.80567741394,
    6224.256128311157,
    7157.11431312561,
    7201.072872161865]

# Objectives results
objectives_1stSolution = [
    39.50361808252076,
    57.83661482379392,
    64.91037122564715,
    124.8183372157811,
    187.68773168539238,
    109.29472585051663,
    289.11165656392,
    367.39009072838877,
    470.8415002948603,
    551.7259686320184,
    469.1066575737234,
    261.50408936881604,
    660.23239584534523,
    1027.2057230572934,
    864.971618409769]

objectives_bestSolution = [
    22.55508482497172,
    29.196976063608044,
    33.24195613211528,
    35.864904130331205,
    49.56598818246004,
    42.868446885983374,
    73.41027767452577,
    75.80273030509582,
    106.75627022933315,
    112.66963723060917,
    93.06594035811467,
    111.57691844865784,
    135.09672962848188,
    201.22478741663983,
    471.76961441535343]

# %%
plt.plot(range(1,16), times_1stSolution, label="Runtime 1st solution")
plt.plot(range(1,16), times_bestSolution, label="Runtime best solution")
plt.title("Runtime comparison between 1st and best solution")
plt.legend(loc='best')
plt.xlabel("Instances")
plt.ylabel("Runtime (seconds)")
plt.savefig("FigXX-RuntimeComparison.png", bbox_inches='tight', dpi=600)
plt.show()

# %%
plt.plot(range(1,16), objectives_1stSolution, label="Objective 1st solution")
plt.plot(range(1,16), objectives_bestSolution, label="Objective best solution")
plt.title("Objective comparison between 1st and best solution")
plt.legend(loc='best')
plt.xlabel("Instances")
plt.ylabel("Objective €")
plt.savefig("FigXX-ObjectiveComparison.png", bbox_inches='tight', dpi=600)
plt.show()

# %%



