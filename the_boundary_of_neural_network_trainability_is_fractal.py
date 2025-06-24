import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib import cm
from scipy import ndimage
import time
import datetime
import pickle
import shutil
import porespy as ps

import debugpy
debugpy.listen(5679)
debugpy.wait_for_client()

interactive_gui = False
width = 8
depth = 2
dataset_param_multiple = 1.
minibatch_size = None
target_dim = 3
nonlinearity = 'tanh'
phase_space = 'lr_vs_lr'
readout = 'loss'
opt_steps = 100
default_resolution = 512
dpi = 100
figsize = (8,8)

def canonical_name():
  """
  turn hyperparameters in the previous cell into a canonical base filename to
  use for this experimental condition
  """
  return f'zoom_sequence_width-{width}_depth-{depth}_datasetparamratio-{dataset_param_multiple}_minibatch-{minibatch_size}_nonlinearity-{nonlinearity}_phasespace-{phase_space}{"_readout-probe_point" if readout == "probe_point" else ""}'

image_history = []

# def perturb_weights(theta, rng, epsilon=1e-5):
    # keys = jax.random.split(rng, len(theta))
    # keys_list = list(keys)
    # perturbed_theta = jax.tree_map(lambda w, k: w + epsilon * jax.random.normal(k, w.shape), theta, keys_list)
    # return perturbed_theta
# 
def perturb_weights(theta, rng, depth, epsilon=1e-5):
    rng = jax.random.split(rng, depth)
    _perturbed_theta = []
    for i, t in enumerate(theta):
      _perturbed_theta.append(t + epsilon * jax.random.normal(rng[i], t.shape))
    return _perturbed_theta


# def compute_lyapunov_exponent(original_output, perturbed_output, epsilon, t):
#     divergence = jnp.abs(perturbed_output - original_output)
#     return jnp.log(divergence / epsilon) / t

def compute_lyapunov_exponent(original_output, perturbed_output, epsilon, step):
    # Small constant to prevent log(0)
    small_constant = 1e-10

    divergence = jnp.abs(perturbed_output - original_output)
    adjusted_divergence = jax.tree_map(lambda d: d + small_constant, divergence)

    # Compute Lyapunov exponent with adjusted divergence
    lyapunov_exponents = jax.tree_map(lambda d: jnp.log(d / epsilon) / step, adjusted_divergence)

    ## check if any element is nan
    if jnp.any(jnp.isnan(lyapunov_exponents)):
        print("lyapunov_exponents", lyapunov_exponents)

    # mean_lyapunov_exponent = jnp.mean(lyapunov_exponents, axis=(1, 2))

    return lyapunov_exponents.max()


def net(theta, X):
  # mean field parameterization
  for W in theta:
    Z = jnp.dot(X, W)/jnp.sqrt(W.shape[0])
    if nonlinearity == 'tanh':
      X = jax.nn.tanh(Z*jnp.sqrt(2))
    elif nonlinearity == 'relu':
      X = jax.nn.relu(Z)*jnp.sqrt(2)
    elif nonlinearity == 'identity':
      X = Z
    else:
      assert False
  return Z /jnp.sqrt(W.shape[0])

def init(rng, width, depth):
  rng = jax.random.split(rng, depth)
  theta = []
  out_width = width
  for i in range(depth):
    if i == depth-1:
      out_width = target_dim
    W = jax.random.normal(rng[i], (width, out_width))
    theta.append(W)
  return theta

def loss(theta, X, Y):
  Z = net(theta, X)
  return jnp.mean((Z - Y)**2), Z

def hparams_f(hparams, theta):
  """
  convert hyperparameters into a learning rate for each layer in the network
  """
  lr = []
  for i, t in enumerate(theta):
    lr.append(hparams[i%len(hparams)])
  return lr

def train_step(rng, theta, hparams, X, Y):
  if phase_space == 'lr_vs_lr':
    learning_rates = hparams_f(hparams, theta)
  elif phase_space == 'paraminit_vs_lr':
    new_theta = []
    for i, t in enumerate(theta):
      if i == 0:
        t += hparams[0]
      new_theta += [t]
    theta = new_theta
    learning_rates = hparams_f([hparams[1]], theta)
  else:
    assert False, f'invalid phase space {phase_space}'

  if minibatch_size is None:
    # _loss, _grad = jax.value_and_grad(loss)(theta, X, Y)
    (_loss, Z), _grad = jax.value_and_grad(loss, has_aux=True)(theta, X, Y)
  else:
    idx = jax.random.randint(rng, (minibatch_size,), 0, X.shape[0])
    _X = X[idx]
    _Y = Y[idx]
    # evaluate loss on full batch for smooth visualization
    (_loss, Z), _grad = jax.value_and_grad(loss, has_aux=True)(theta, X, Y)
    _grad = jax.grad(loss, has_aux=True)(theta, _X, _Y)[1]

  if readout == 'probe_point':
    srng = jax.random.PRNGKey(-1234567)
    X_probe = jax.random.normal(srng, (1, X.shape[1]))
    tracked_value = net(theta, X_probe).ravel()[0]
  elif readout == 'loss':
    tracked_value = _loss

  return jax.tree_map(lambda t, g, lr: t - lr*g, theta, _grad, learning_rates), tracked_value, Z

train_step_lrvmap = jax.jit(
    jax.vmap(train_step, in_axes=(None, 0, 0, None, None), out_axes=(0, 0, 0)))

# train_step_lrvmap = train_step()


def train(theta, perturbed_theta, hparams, X, Y, num_steps, outer_batch_size=50000):

  bs = hparams.shape[0]
  epsilon = 1e-5

  if bs > outer_batch_size:
      # Split up the hyperparameter search grid if we would run out of memory
      results1 = train(theta, perturbed_theta,  hparams[:bs//2], X, Y, num_steps)
      results2 = train(theta, perturbed_theta, hparams[bs//2:], X, Y, num_steps)
      # Concatenate results for regular and perturbed models separately
      losses = jnp.concatenate((results1[0], results2[0]), axis=0)
      perturbed_losses = jnp.concatenate((results1[1], results2[1]), axis=0)
      lyapunov_exponents = jnp.concatenate((results1[2], results2[2]), axis=0)
      return losses, perturbed_losses, lyapunov_exponents

  rng = jax.random.PRNGKey(42)
  perturb_rng, rng = jax.random.split(rng) 
  rng = jax.random.split(rng, num_steps)
  losses = []
  perturbed_losses = []
  lyapunov_exponents = []
  _theta = jax.tree_map(lambda u: jnp.tile(u, (bs,) + (1,)*len(u.shape)), theta)
  _perturbed_theta = jax.tree_map(lambda u: jnp.tile(u, (bs,) + (1,)*len(u.shape)), perturbed_theta)
  # _perturbed_theta = perturb_weights(_theta, perturb_rng, epsilon=epsilon)
  perturb_rng = jax.random.split(perturb_rng, num_steps)
  for i, (_rng, _p_rng) in enumerate(zip(rng, perturb_rng)):
    if i==64:
      print('Time to debug')
    _theta, _loss, _output = train_step_lrvmap(_rng, _theta, hparams, X, Y)
    _perturbed_theta, _perturbed_loss, _perturbed_output = train_step_lrvmap(_p_rng, _perturbed_theta, hparams, X, Y)
    lyapunov_exponents.append(compute_lyapunov_exponent(_output, _perturbed_output, epsilon, i+1))
    losses.append(_loss)
    perturbed_losses.append(_perturbed_loss)

  return convergence_measure_vmap(jnp.stack(losses, axis=-1)), convergence_measure_vmap(jnp.stack(perturbed_losses, axis=-1)), jnp.stack(lyapunov_exponents, axis=-1)

# @jax.jit
def convergence_measure(v, max_val = 1e6):
  """
  turn the training trajectory into a single number which looks pretty in an
  image
  """

  fin = jnp.isfinite(v)
  v = v*fin + max_val*(1-fin)

  if readout == 'probe_point':
    # return the final value of the function, at the end of training
    return v[-1]

  assert readout == 'loss', 'invalid readout type'

  v /= v[0]
  exceeds = (v > max_val)
  v = v*(1-exceeds) + max_val*exceeds

  converged = (jnp.mean(v[-20:]) < 1) # average over any oscillatory behavior
  return jnp.where(converged, -jnp.sum(v), jnp.sum(1/v))

convergence_measure_vmap = jax.jit(jax.vmap(convergence_measure, in_axes=(0,), out_axes=0))

# lyapunov_exponents_vmap = jax.jit(jax.vmap(compute_lyapunov_exponent, in_axes=(0, 0, 0, 0), out_axes=0))


def gen_img(mnmx, resolution=None):
  """
  generate an image of the hyperparameter landscape,
  for a range of hyperparameter values specified by mnmx
  """

  if resolution is None:
    resolution = default_resolution

  mn1, mx1, mn2, mx2 = mnmx
  rng = jax.random.PRNGKey(0)
  rng, srng = jax.random.split(rng)
  theta = init(srng, width, depth)
  perturbed_theta = perturb_weights(theta, rng, depth)
  # perturb_rng, rng = jax.random.split(rng) 
  # rng = jax.random.split(rng, num_steps)

  n_params = jnp.sum(jnp.array(jax.tree_map(lambda u: u.size, theta)))

  rng, srng = jax.random.split(rng)

  if dataset_param_multiple == 'single_datapoint':
    batch_size = 1
  else:
    if nonlinearity == 'identity':
      batch_size = int(width*dataset_param_multiple)
    else:
      batch_size = int(n_params*dataset_param_multiple/target_dim)

  X = jax.random.normal(srng, (batch_size, width))
  rng, srng = jax.random.split(rng)
  Y = jax.random.normal(srng, (batch_size, target_dim))

  gg1 = jnp.logspace(mn1, mx1, resolution)
  gg2 = jnp.logspace(mn2, mx2, resolution)
  lr0, lr1 = jnp.meshgrid(gg2, gg1)
  lr = jnp.stack([lr0.ravel(), lr1.ravel()], axis=-1)

  V, V_perturb, lyapunov_exponents = train(theta, perturbed_theta, lr, X, Y, opt_steps)

  return V.reshape((resolution, resolution)), V_perturb.reshape((resolution, resolution)), lyapunov_exponents.reshape((resolution, resolution))


def extract_edges(X):
  """
  define edges as sign changes in the scalar representing convergence or
  divergence rate -- on one side of the edge training converges,
  while on the other side of the edge training diverges
  """

  Y = jnp.stack((X[1:,1:], X[:-1,1:], X[1:,:-1], X[:-1,:-1]), axis=-1)
  Z = jnp.sign(jnp.max(Y, axis=-1)*jnp.min(Y, axis=-1))
  return Z<0

def estimate_fractal_dimension(hist_video, show_plot=True):
  edges = [extract_edges(U[0]) for U in hist_video]
  box_counts = [ps.metrics.boxcount(U) for U in edges]
  all_images = np.concatenate([bc.slope for bc in box_counts])

  if show_plot:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_xlabel('box edge length')
    ax1.set_ylabel('number of boxes spanning phases')
    ax2.set_xlabel('box edge length')
    ax2.set_ylabel('image')
    ax2.set_xscale('log')

    for bc in box_counts:
      ax1.plot(bc.size, bc.count,'-o')
      ax2.plot(bc.size, bc.slope,'-o');

  mfd = np.median(all_images)
  print(f'median fractal dimension estimate {mfd}')

  return mfd


def cdf_img(x, x_ref, buffer=0.25):
  """
  rescale x, relative to x_ref (x_ref is often the same as x), to achieve a uniform
  distribution over values with positive and negative intensities, but also to
  preserve the sign of x. This makes for a visualization that shows more
  structure.
  """

  u = jnp.sort(x_ref.ravel())

  if readout == 'probe_point':
    v = jnp.linspace(-1, 1, u.shape[0])
  elif readout == 'loss':
    num_neg = jnp.sum(u<0)
    num_nonneg = u.shape[0] - num_neg
    v = jnp.concatenate((jnp.linspace(-1,-buffer,num_neg), jnp.linspace(buffer,1,num_nonneg)), axis=0)
  else:
    assert False, 'invalid readout'

  y = jnp.interp(x, u, v)
  return -y


def truncate_sci_notation(numbers):
  """
  keeping enough significant digits that the
  numbers disagree in four digits
  """

  # Convert numbers to scientific notation
  n1_sci, n2_sci = "{:.15e}".format(numbers[0]), "{:.15e}".format(numbers[1])

  # Extract the significant parts and exponents
  sig_n1, exp_n1 = n1_sci.split('e')
  sig_n2, exp_n2 = n2_sci.split('e')

  # Find the first position at which they disagree
  min_len = min(len(sig_n1), len(sig_n2))
  truncate_index = min_len

  for i in range(min_len):
      if (sig_n1[i] != sig_n2[i]) or (exp_n1 != exp_n2):
          # +4 accounts for 4 digits after the first disagreement
          truncate_index = i + 4
          if i == 0:
            truncate_index += 1 # Account for decimal point
          break

  exp_n1 = exp_n1[0] + exp_n1[2]
  exp_n2 = exp_n2[0] + exp_n2[2]
  if (exp_n1 == "+00") and (exp_n2 == "+00"):
    # don't bother with scientific notation if exponent is 0
    return [sig_n1[:truncate_index], sig_n2[:truncate_index]]

  # Truncate and reconstruct the scientific notation
  truncated_n1 = "{}e{}".format(sig_n1[:truncate_index], exp_n1)
  truncated_n2 = "{}e{}".format(sig_n2[:truncate_index], exp_n2)

  return [truncated_n1, truncated_n2]

def tickslabels(mnmx):
  return mnmx, truncate_sci_notation(10.**np.array(mnmx))



cids = []
click_event = [None]

def onclick(event):
  click_event[0] = (event.xdata, event.ydata)

def onrelease(event, fig, im, rect, mnmx, img, recalculate_image=True):
  if click_event[0] is None:
    return

  e0 = [click_event[0][0], event.xdata]
  e1 = [click_event[0][1], event.ydata]

  for v in e0+e1:
    if v is None:
      return

  newmnmx = [np.min(e1), np.max(e1), np.min(e0), np.max(e0)]

  min_w = (mnmx[1] - mnmx[0])/20
  if newmnmx[1] - newmnmx[0] < min_w:
    c = (newmnmx[1] + newmnmx[0])/2.
    newmnmx[0] = c - min_w/2
    newmnmx[1] = c + min_w/2
  min_w = (mnmx[3] - mnmx[2])/20
  if newmnmx[1] - newmnmx[0] < min_w:
    c = (newmnmx[3] + newmnmx[2])/2.
    newmnmx[2] = c - min_w/2
    newmnmx[3] = c + min_w/2

  for v in newmnmx:
    if v is None:
      return
  plot_img(img, mnmx, newmnmx, fig=fig, im=im, rect=rect)
  plt.draw()

  if recalculate_image:
    click_event[0] = None
    mnmx = newmnmx
    img, img_pertutb = gen_img(mnmx)
    plot_img(img, mnmx, None, fig=fig, im=im, rect=rect)
    plot_img(img_pertutb, mnmx, None, fig=fig, im=im, rect=rect)

def plot_img(image, mnmx, newmnmx=None, fig=None, im=None, rect=None,
             handler=True, savename=None,
             reference_scale=None,
             cmap='Spectral',
             title=""
             ):
  mn1, mx1, mn2, mx2 = mnmx

  if reference_scale is None:
    reference_scale = image

  image = cdf_img(image, reference_scale)

  ax1 = None
  if fig is None:
    fig, (ax1) = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax1.imshow(image,
                    extent=[mn2, mx2, mn1, mx1],
                    origin='lower',
                    vmin=-1, vmax=1,
                    cmap=cmap,
                    aspect='auto',
                    interpolation='nearest'
                    )
    if title is None:
      batch_text = "full batch" if minibatch_size is None else "minibatch"
      if dataset_param_multiple == 'single_datapoint':
        batch_text = 'single training point'
      title = f'Trainability dependence on {"per-layer learning rates" if phase_space == "lr_vs_lr" else "parameter initialization and learning rate"}\n1 hidden layer, {nonlinearity}, {batch_text}'
    if not title == "":
      plt.title(title)
    if phase_space == 'lr_vs_lr':
      ax1.set_ylabel('Output layer learning rate')
      ax1.set_xlabel('Input layer learning rate')
    elif phase_space == 'paraminit_vs_lr':
      ax1.set_ylabel('Learning rate')
      ax1.set_xlabel('Input layer weight offset')

    rect = patches.Rectangle((mn2, mn1), mx2-mn2, mx1-mn1, linewidth=1, edgecolor='r', facecolor='none')
    ax1.add_patch(rect)

  im.set_extent([mn2, mx2, mn1, mx1])
  im.set_data(image)

  # Set the new tick positions on the x-axis
  aaxx = plt.gca()
  aaxx.set_xticks(*tickslabels([mn2, mx2]))
  aaxx.set_yticks(*tickslabels([mn1, mx1]), rotation=90)

  labels = aaxx.get_xticklabels()
  labels[0].set_horizontalalignment('left')
  labels[1].set_horizontalalignment('right')
  labels = aaxx.get_yticklabels()
  labels[0].set_verticalalignment('bottom')
  labels[1].set_verticalalignment('top')

  if handler and (newmnmx is None):
    image_history.append((image, mnmx))

  if newmnmx:
    mn1, mx1, mn2, mx2 = newmnmx
  rect.set_xy((mn2, mn1))
  rect.set_width(mx2-mn2)
  rect.set_height(mx1-mn1)

  if handler:
    while len(cids) > 0:
      fig.canvas.mpl_disconnect(cids.pop())

    def onrelease_partial(event):
      return onrelease(event, fig, im, rect, mnmx, img)
    def onmotion_partial(event):
      return onrelease(event, fig, im, rect, mnmx, img, recalculate_image=False)

    cids.append(fig.canvas.mpl_connect('button_press_event', onclick))
    cids.append(fig.canvas.mpl_connect('button_release_event', onrelease_partial))
    # cids.append(fig.canvas.mpl_connect('motion_notify_event', onmotion_partial))

  plt.tight_layout()

  plt.draw()

  if savename:
    plt.savefig(savename)

  return fig, ax1, im


def zoom_out_sequence(hist_final, growth_factor=2., max_scale=6):
  """
  generate a sequence of (image, bounds) zooming out from the (image, bounds) in hist_final
  """

  image, mnmx = hist_final

  cT = np.array([(mnmx[0] + mnmx[1])/2., (mnmx[2] + mnmx[3])/2.])
  wT = np.array([mnmx[1] - mnmx[0], mnmx[3] - mnmx[2]])

  hist = [(image, mnmx)]
  w_scale = 1.
  while np.min(wT * w_scale) < max_scale:
    w_scale *= 2
    mnmx = [
        cT[0] - w_scale * wT[0]/2.,
        cT[0] + w_scale * wT[0]/2.,
        cT[1] - w_scale * wT[1]/2.,
        cT[1] + w_scale * wT[1]/2.,
    ]
    hist.insert(0, (np.zeros((2,2)), mnmx))

  return hist

def increase_resolution(history, target_res):
  """
  Increase the resolution of images of a fractal landscape that we've already
  generated.

  Find the first entry in history with resolution below target_res, and increase
  its resolution. If all images are already at least the target resolution,
  return False.
  """

  new_h = []
  for ii in range(len(history)):
    h = history[ii]
    image, mnmx = h
    if image.shape[0] < target_res:
      current_time = datetime.datetime.now()
      print( f"increasing resolution of {ii} / {len(history)} at {current_time}, current resolution is {image.shape}")
      image = gen_img(mnmx, resolution=target_res)
      history[ii] = (image, mnmx)
      return True
  return False


def interpolate_history(hist1, hist2, alpha):
  """
  get the mnmx (hyperparameter bounding box) value for a fraction alpha between
  two images
  """

  _, mnmx1 = hist1
  _, mnmx2 = hist2

  if alpha == 0:
    # avoid NaNs on very last frame
    return mnmx1

  w1 = np.array([mnmx1[1] - mnmx1[0], mnmx1[3] - mnmx1[2]])
  w2 = np.array([mnmx2[1] - mnmx2[0], mnmx2[3] - mnmx2[2]])
  c1 = np.array([(mnmx1[0] + mnmx1[1])/2, (mnmx1[2] + mnmx1[3])/2])
  c2 = np.array([(mnmx2[0] + mnmx2[1])/2, (mnmx2[2] + mnmx2[3])/2])

  gamma = np.exp((1-alpha)*0 + alpha*np.log(w2/w1))

  # ct = cstar + (c1 - cstar)*gamma
  # c1 = cstar + (c1 - cstar)*1
  # c2 = cstar + (c1 - cstar)*w2/w1
  cstar = (c2 - c1*w2/w1) / (1 - w2 / w1)

  ct = cstar + (c1 - cstar)*gamma
  hwt = gamma*w1

  return [ct[0] - hwt[0]/2, ct[0] + hwt[0]/2, ct[1] - hwt[1]/2, ct[1] + hwt[1]/2]


def em(extent_rev):
  return [extent_rev[2], extent_rev[3], extent_rev[0], extent_rev[1]]

def make_animator(history, timesteps_per_transition=60, reference_scale=None, cmap='Spectral'):

  fig, ax, im1 = plot_img(history[0][0], history[0][1], newmnmx=None,
                          handler=False, reference_scale=reference_scale, cmap=cmap)
  im2 = ax.imshow(
      jnp.zeros_like(history[1][0]), extent=em(history[1][1]), origin='lower',
      vmin = -1, vmax = 1,
      cmap=cmap,
      aspect='auto',
      interpolation='nearest'
      )

  im3 = ax.imshow(
      jnp.zeros_like(history[1][0]), extent=em(history[1][1]), origin='lower',
      vmin = -1, vmax = 1,
      cmap=cmap,
      aspect='auto',
      interpolation='nearest'
      )

  def animate(n):
    hist_index = n // timesteps_per_transition
    alpha = (n % timesteps_per_transition) / timesteps_per_transition

    hist1 = history[hist_index]
    if hist_index >= len(history)-1:
      hist2 = hist1 # very last frame
    else:
      hist2 = history[hist_index+1]
    if hist_index >= len(history)-2:
      hist3 = hist2 # very last frame
    else:
      hist3 = history[hist_index+2]

    lims = interpolate_history(hist1, hist2, alpha)

    # interpolation scheme for image restretch / colormap
    alpha_area = jnp.sin(alpha*np.pi/2)**2

    print(f'frame {n} / {timesteps_per_transition*len(history)}, zoom step {hist_index} / {len(history)}', end='\r', flush=True)

    img_1 = (1-alpha_area)*cdf_img(hist1[0], hist1[0]) + alpha_area*cdf_img(hist1[0], hist2[0])
    img_2 = (1-alpha_area)*cdf_img(hist2[0], hist1[0]) + alpha_area*cdf_img(hist2[0], hist2[0])
    img_3 = (1-alpha_area)*cdf_img(hist3[0], hist1[0]) + alpha_area*cdf_img(hist3[0], hist2[0])

    im1.set_data(img_1)
    im1.set_extent(em(hist1[1]))
    im2.set_data(img_2)
    im2.set_extent(em(hist2[1]))
    im3.set_data(img_3)
    im3.set_extent(em(hist3[1]))
    im3.set_alpha(alpha)

    ax.set_ylim(lims[0], lims[1])
    ax.set_xlim(lims[2], lims[3])

    # Set the new tick positions
    ax.set_xticks(*tickslabels([lims[2], lims[3]]))
    ax.set_yticks(*tickslabels([lims[0], lims[1]]), rotation=90)

    labels = ax.get_xticklabels()
    labels[0].set_horizontalalignment('left')
    labels[1].set_horizontalalignment('right')
    labels = ax.get_yticklabels()
    labels[0].set_verticalalignment('bottom')
    labels[1].set_verticalalignment('top')

    return fig,

  anim = animation.FuncAnimation(fig,animate,frames=timesteps_per_transition*(len(history)-1)+1, repeat=False)
  return anim


plt.close('all')
plt.ion()

mnmx = [-3, 6, -3, 6]
img, img_perturb, lyapunov_exponents = gen_img(mnmx)
plot_img(img, mnmx, None)
plot_img(img_perturb, mnmx, None)
plt.plot(lyapunov_exponents)
plt.show()


plt.close('all')

for ii, impair in enumerate(image_history):
  image, mnmx = impair
  newmnmx = None
  if ii < len(image_history)-1:
    newmnmx = image_history[ii+1][1]
  # fname = f'fractal_train_width{width}_depth{depth}_datasetparamratio{dataset_param_multiple}_minibatch{minibatch_size}_nonlinearity{nonlinearity}_step{ii}_coords{mnmx}.pdf'
  fname = None
  # print(ii, mnmx)
  plot_img(image, mnmx, newmnmx=newmnmx, handler=False, savename=fname)
  plt.title(mnmx)
  plt.tight_layout()
  # files.download(fname)

hist_video = zoom_out_sequence(image_history[-1], growth_factor=2.)

# each call to increase_resolution increases the resolution of one image and
# returns True, or returns False if all images are at or exceed the target resolution
while increase_resolution(hist_video, 2048):
  with open(f'/home/omkar/Desktop/EEE560/Project/output/{canonical_name()}.pickle', 'wb') as handle:
      pickle.dump(hist_video, handle, protocol=pickle.HIGHEST_PROTOCOL)

ts = 30
anim = make_animator(hist_video, timesteps_per_transition=ts*2)
anim.save(f'/home/omkar/Desktop/EEE560/Project/output/{canonical_name()}.mp4',fps=ts, dpi=dpi)
plt.close('all')


def load_and_generate(canonical=None, max_frame=999999):
  global dpi
  global figsize

  dpi = 1000
  figsize = (5.2,4.5)

  assert not interactive_gui, "Set interactive_gui to False in the first cell, and restart the colab kernel"

  if canonical is None:
    canonical = canonical_name()

  print(f'generating images and movie for {canonical}')

  with open(f'/home/omkar/Desktop/EEE560/Project/output/{canonical}.pickle', 'rb') as handle:
    hist_video = pickle.load(handle)

  hist_video = hist_video[:max_frame]

  # make sure we are at full resolution
  # each call to increase_resolution increases the resolution of one image and returns True, or returns False if all images are at or exceed the targer resolution
  while increase_resolution(hist_video, 4096):
    with open(f'/content/drive/MyDrive/fractal/{canonical}.pickle', 'wb') as handle:
        pickle.dump(hist_video, handle, protocol=pickle.HIGHEST_PROTOCOL)

  for ii, impair in enumerate(hist_video):
    img, mnmx = impair
    fname = f'/content/drive/MyDrive/fractal/{canonical}_step-{ii}.pdf'
    plot_img(img, mnmx, handler=False, savename=fname, title="")
    plt.close('all')

  estimate_fractal_dimension(hist_video)
  plt.close('all')

  # dpi of 1500 might seem excessive, but we need a very high resolution,
  # otherwise we get ugly transcoding errors when we upload to vimeo or youtube
  dpi = 1500
  figsize = (5.5,5.5)
  ts = 30
  anim = make_animator(hist_video, timesteps_per_transition=int(ts*1.5))
  fname = f'{canonical}.mp4'
  writer=animation.FFMpegWriter(bitrate=200000, codec='hevc', fps=ts)
  anim.save(fname, dpi=dpi, writer=writer)
  shutil.copyfile(fname, f'/content/drive/MyDrive/fractal/{fname}')
  plt.close('all')

"""
## batch size 1
"""

width = 16
depth = 2
dataset_param_multiple = 'single_datapoint'
minibatch_size = None
target_dim = 1
nonlinearity = 'tanh'
phase_space = 'lr_vs_lr'
readout = 'loss'

load_and_generate()

# zoom_sequence_width-16_depth-2_datasetparamratio-single_datapoint_minibatch-None_nonlinearity-tanh_phasespace-lr_vs_lr.pickle

# %%
"""
## tanh full batch initialization shift vs learning rate
"""

# %%
width = 16
depth = 2
dataset_param_multiple = 1.
minibatch_size = None
target_dim = 1
nonlinearity = 'tanh'
phase_space = 'paraminit_vs_lr'
readout = 'loss'

load_and_generate()

# %%
"""
## linear full batch
"""

# %%
width = 16
depth = 2
dataset_param_multiple = 1.
minibatch_size = None
target_dim = 1
nonlinearity = 'identity'
phase_space = 'lr_vs_lr'
readout = 'loss'

load_and_generate()

# %%
"""
## relu full batch
"""

# %%
width = 16
depth = 2
dataset_param_multiple = 1.
minibatch_size = None
target_dim = 1
nonlinearity = 'relu'
phase_space = 'lr_vs_lr'
readout = 'loss'

load_and_generate(max_frame=52)

# %%
"""
## tanh full batch

"""

# %%
width = 16
depth = 2
dataset_param_multiple = 1.
minibatch_size = None
target_dim = 1
nonlinearity = 'tanh'
phase_space = 'lr_vs_lr'
readout = 'loss'

load_and_generate(max_frame=49)

# %%
"""
## tanh minibatch
"""

# %%
width = 16
depth = 2
dataset_param_multiple = 1.
minibatch_size = 16
target_dim = 1
nonlinearity = 'tanh'
phase_space = 'lr_vs_lr'
readout = 'loss'

load_and_generate()
