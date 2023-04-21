import numpy as np
import matplotlib.pyplot as plt

# Load actual data
d = np.load('data/FalknerSkan_n0.08.npz')
u_actual = d['u'].T
v_actual = d['v'].T
p_actual = d['p'].T

# Load predicted data
res = np.load('pred/res_FS_20_8_tanh_1000_1206_500.npz')
u_pred = res['pred'][0]
v_pred = res['pred'][1]
p_pred = res['pred'][2]

# Calculate the errors
u_err = np.abs(u_actual - u_pred)
v_err = np.abs(v_actual - v_pred)
p_err = np.abs(p_actual - p_pred)

# Create a figure for u
fig_u, axs_u = plt.subplots(1, 3, figsize=(14, 4))
fig_u.suptitle('u')

# Plot u actual and predicted
axs_u[0].contourf(u_actual)
axs_u[0].set_title('u_actual')
axs_u[1].contourf(u_pred)
axs_u[1].set_title('u_predicted')
axs_u[2].contourf(np.abs(u_actual - u_pred))
axs_u[2].set_title('u_error')

# Add colorbars to each subplot
for ax in axs_u.flat:
    fig_u.colorbar(ax.contourf(np.random.rand(10,10)), ax=ax)

# Create a figure for v
fig_v, axs_v = plt.subplots(1, 3, figsize=(14, 4))
fig_v.suptitle('v')

# Plot v actual and predicted
axs_v[0].contourf(v_actual)
axs_v[0].set_title('v_actual')
axs_v[1].contourf(v_pred)
axs_v[1].set_title('v_predicted')
axs_v[2].contourf(np.abs(v_actual - v_pred))
axs_v[2].set_title('v_error')

# Add colorbars to each subplot
for ax in axs_v.flat:
    fig_v.colorbar(ax.contourf(np.random.rand(10,10)), ax=ax)

# Create a figure for p
fig_p, axs_p = plt.subplots(1, 3, figsize=(14, 4))
fig_p.suptitle('p')

# Plot p actual and predicted
axs_p[0].contourf(p_actual)
axs_p[0].set_title('p_actual')
axs_p[1].contourf(p_pred)
axs_p[1].set_title('p_predicted')
axs_p[2].contourf(np.abs(p_actual - p_pred))
axs_p[2].set_title('p_error')

# Add colorbars to each subplot
for ax in axs_p.flat:
    fig_p.colorbar(ax.contourf(np.random.rand(10,10)), ax=ax)

# Show the plots
plt.show()
