import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

matplotlib.rc('animation', html='jshtml')


def plot_animation(frames, repeat=False, interval=40):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    anim = animation.FuncAnimation(
        fig=fig,
        func=update_scene,
        fargs=(frames, patch),
        frames=len(frames),
        repeat=repeat,
        interval=interval
    )
    plt.close()
    return anim


def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch
