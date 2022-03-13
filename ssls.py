# -*- coding: utf-8 -*-
# SSLS - Star System Lightcurve Simulator
# Body motions Inspired by https://colab.research.google.com/drive/1YKjSs8_giaZVrUKDhWLnUAfebuLTC-A5
# Needs ffmpeg to convert the data into a video. Download ffmpeg from https://www.ffmpeg.org/download.html. Extract the zip file and add "<yourdriveandpath>\FFmpeg\bin" to Environment Variable PATH.

import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt
import math
import numpy as np
import time


# Parameters
video_file = "ssls_TIC_470710327.mp4"

Frames = 2500       # number of iterations/frames. Proportional to this program's run time and to the lenght of the animation.
FPS = 30           # frames per second in video. Proportional to the velocity of the objects in the animation. Inverse proportional to lenght of video.
DT = 1            # time difference for one iteration [s]. Proportional to the velocity of the objects in the animation. 3600 seems a good value for precise enough orbit calculations.
Sampling_rate = 2000  # Calculating the physics is much faster than animating it. Therefore only 1/Sampling_rate of the calculated iterations is used as frame in the animation.
Iterations = Frames * Sampling_rate

G  = 6.67430e-11     # gravitational constant [m**3/kg/s**2] +/- 0.00015
AU = 1.495978707e11  # astronomical unit [m]
SR = 6.96342e8       # sun radius [m]
SM = 1.98847e30      # sun mass [kg] +/- 0.00007
SL = 3.83e26         # solar luminosity [W]
ER = 6.378135e6      # earth radius [m]
EM = 5.9720e24       # earth mass [kg]
EV = 2.97852e4       # earth orbital velocity [m/s] (if orbit was circular)

Scale_ecl = 0.9 * AU      # height of eclipse view plotting window in meters. Middle of window is (0.0, 0.0)
StarScale_ecl = 1.0       # animate stars with StarScale_ecl times enlarged radius.
PlanetScale_ecl = 100.0   # animate planets with PlanetScale_ecl times enlarged radius.
Scale_top = 0.9 * AU      # height of top view plotting window in meters. Middle of window is (0.0, 0.0)
StarScale_top = 1.0       # animate stars with StarScale_top times enlarged radius.
PlanetScale_top = 100.0   # animate planets with PlanetScale_top times enlarged radius.


def todo():
    print("")
    print("3-Star-System aus dem Paper nachbauen.")
    print("Funktioniert alles auch für mehrere Sonnen und mehrere Planeten?")
    print("")
    print("Parameter für Stern, der einen Faktor angibt, um den das Sternzentrum heller ist als der Sternrand. Dann kann ich bei Eclipse die Helligkeit an der verdeckten Sternstelle interpolieren.")
    print("Funktion, die die Transitzeitpunkte ermittelt und speichert und Umlaufzeit-Variationen aufgrund zusätzlicher Körper erfasst.")
    print("")
    print("Parameter und Options in ein File auslagern? Oder zumindest die ganzen Konstanten in init_plot als globale Parameter deklarieren.")
    print("")


class Body():
    def __init__(self, name="no_name", mass=1.0, radius=1.0, brightness=0.0, luminosity=0.0, startposition=np.array([0.0, 0.0, 0.0]), velocity=np.array([0.0, 0.0, 0.0]), color=(0.0, 0.0, 0.0), extrascale_top=1.0, extrascale_ecl=1.0):
        self.name = name                                    # name
        self.mass = mass                                    # [kg]
        self.radius = radius                                # [m]
        self.area_2d = np.pi * radius**2                    # [m**2]
        self.brightness = brightness                        # luminosity per (apparent) area [W/m**2]
        self.luminosity = luminosity                        # [W]
        self.positions = np.zeros((Iterations,3))           # position for each frame
        self.positions[0] = startposition                   # [m] initial position
        self.velocity = velocity                            # [m/s] (np.array)
        self.color = color                                  # (R, G, B)  each between 0 and 1
        self.circle_top = matplotlib.patches.Circle((0, 0), radius * extrascale_top / Scale_top)  # matplotlib patch for top view
        self.circle_ecl = matplotlib.patches.Circle((0, 0), radius * extrascale_ecl / Scale_ecl)  # matplotlib patch for eclipsed view

        if self.luminosity + self.brightness > 0: # One of them is >0, so it's a star!
            if self.luminosity > 0:
                self.brightness = self.luminosity / self.area_2d
            else:
                self.luminosity = self.brightness * self.area_2d


    def eclipsed_by(self, body, frame):
        if body.positions[frame][1] < self.positions[frame][1]:  # body nearer to viewpoint than self?
            d = distance_2d_ecl(body, self, frame)
            if d < self.radius + body.radius:  # does body eclipse self?
                if d < self.radius - body.radius:
                    return body.area_2d  # full eclipse!
                else:  # partial eclipse!
                    # print(f'd: {d:6.3e}  rs: {self.radius:6.3e}  rb: {body.radius:6.3e}')
                    h = self.radius + body.radius - d
                    angle = 2 * math.acos(1-h/body.radius)
                    return body.radius**2 * (angle - math.sin(angle)) / 2  # these simplified formulas assume that self.radius is much bigger than body.radius
            else:
                return 0.0  # no eclipse!
        else:
            return 0.0  # reverse eclipse: self eclipses body!


def init_bodies(bodies):
    bodies.append(Body(name="TIC 470710327 A",
                       mass=6.5*SM, radius=2.0*SR,
                       startposition=np.array([0, 3620798086, 0]), velocity=np.array([-235000, -88820, -12688.0]),
                       luminosity=2**3.14*SL, color=(0.99, 0.99, 0.01),
                       extrascale_ecl=StarScale_ecl, extrascale_top=StarScale_top))
    bodies.append(Body(name="TIC 470710327 B",
                       mass=5.9*SM, radius=1.5*SR,
                       startposition=np.array([0, -3620798086, 0]), velocity=np.array([235000, -88820, -12688.0]),
                       luminosity=2**3.08*SL, color=(0.01, 0.01, 0.99),
                       extrascale_ecl=StarScale_ecl, extrascale_top=StarScale_top))
    # bodies.append(Body(name="TIC 470710327 AohneC",
    #                    mass=6.5*SM, radius=2.0*SR,
    #                    startposition=np.array([0.0, 3620798086, 0.0]), velocity=np.array([-x, 0, 0.0]),
    #                    luminosity=SL, color=(0.99, 0.99, 0.01),
    #                    extrascale_ecl=StarScale_ecl, extrascale_top=StarScale_top))
    # bodies.append(Body(name="TIC 470710327 BohneC",
    #                    mass=5.9*SM, radius=1.5*SR,
    #                    startposition=np.array([0.0, -3620798086, 0.0]), velocity=np.array([x, 0, 0.0]),
    #                    luminosity=SL, color=(0.01, 0.01, 0.99),
    #                    extrascale_ecl=StarScale_ecl, extrascale_top=StarScale_top))
    bodies.append(Body(name="TIC 470710327 C",
                       mass=15.25*SM, radius=3.5*SR,
                       startposition=np.array([123338343881, 0, 0]), velocity=np.array([0, 85118, 12688.0]), #88820
                       luminosity=2**4.79*SL, color=(0.99, 0.01, 0.99),
                       extrascale_ecl=StarScale_ecl, extrascale_top=StarScale_top))
    # bodies.append(Body(name="TIC 470710327 A+B",
    #                    mass=12.4*SM, radius=2.0*SR,
    #                    startposition=np.array([0.0, 0.0, 0.0]), velocity=np.array([0.0, 0.0, 0.0]),
    #                    luminosity=SL, color=(0.99, 0.99, 0.01),
    #                    extrascale_ecl=StarScale_ecl, extrascale_top=StarScale_top))
    # bodies.append(Body(name="demo1-Star1",
    #                    mass=SM, radius=SR,
    #                    startposition=np.array([0.0, 0.0, 0.0]), velocity=np.array([0.0, 0.0, 0.0]),
    #                    luminosity=SL, color=(0.99, 0.99, 0.01),
    #                    extrascale_ecl=StarScale_ecl, extrascale_top=StarScale_top))
    # bodies.append(Body(name="demo1-Star2",
    #                    mass=SM*0.5, radius=SR*0.5,
    #                    startposition=np.array([0.1*AU, -1*AU, 0.0]), velocity=np.array([-10*EV, 0.0, 0.0]),
    #                    luminosity=SL * 0.5, color=(0.90, 0.50, 0.01),
    #                    extrascale_ecl=StarScale_ecl, extrascale_top=StarScale_top))
    # bodies.append(Body(name="Sun", mass=SM, radius=SR, startposition=np.array([0.0, 0.0, 0.0]), velocity=np.array([0.0, 0.0, 0.0]), color=(0.90, 0.90, 0.10), extrascale=StarScale_ecl, luminosity=SL))
    # bodies.append(Body(name="Earth", mass=EM, radius=ER, startposition=np.array([-0.004*AU, -1.0*AU, 0.0]), velocity=np.array([-EV, 0.0, 0.0]), color=(0.20, 0.20, 0.80), extrascale=PlanetScale_ecl))
    # bodies.append(Body(name="Earth", mass=EM, radius=ER, startposition=np.array([1.0 * AU, 0.0 * AU, 0.0 * AU]), velocity=np.array([0.0, -EV, 0.0]), color=(0.20, 0.20, 0.80), extrascale=PlanetScale_ecl))


def init_plot(sampled_lightcurve):
    fig = plt.figure()
    fig.set_figwidth(16)
    fig.set_figheight(8)
    xlim = 1.25
    ylim = 1.0
    fig.set_facecolor("black")  # background color outside of ax_eclipse and ax_lightcurve
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)  # Positions of the subplots edges, as a fraction of the figure width.

    ax_eclipse = plt.subplot2grid(shape=(5, 2), loc=(0, 0), rowspan=4, colspan=1)

    ax_eclipse.set_xlim(-xlim, xlim)
    ax_eclipse.set_ylim(-ylim, ylim)
    ax_eclipse.set_aspect('equal')
    ax_eclipse.set_facecolor("black")  # background color
    # ax_eclipse.get_xaxis().set_visible(False)
    # ax_eclipse.get_yaxis().set_visible(False)

    ax_top = plt.subplot2grid(shape=(5, 2), loc=(0, 1), rowspan=4, colspan=1)
    ax_top.set_xlim(-xlim, xlim)
    ax_top.set_ylim(-ylim, ylim)
    ax_top.set_aspect('equal')
    ax_top.set_facecolor("black")  # background color

    ax_lightcurve = plt.subplot2grid(shape=(5, 1), loc=(4, 0), rowspan=1, colspan=1)
    ax_lightcurve.set_xlim(0, Iterations*DT)
    min = lightcurve.min()
    max = lightcurve.max()
    scope = max - min
    buffer = 0.1 * scope
    ax_lightcurve.set_ylim(min - buffer, max + buffer)
    ax_lightcurve.set_facecolor("black")  # background color
    ax_lightcurve.tick_params(axis='x', colors='grey')  # setting up X-axis tick color to red
    ax_lightcurve.tick_params(axis='y', colors='grey')  # setting up Y-axis tick color to black
    ax_lightcurve.plot(range(0, Iterations * DT, Sampling_rate * DT), sampled_lightcurve, color="white")
    red_dot = matplotlib.patches.Ellipse((0,0), Iterations * DT / 200, scope / 13)  # matplotlib patch
    red_dot.set(zorder=2)
    red_dot.set_color((1,0,0))  # red
    ax_lightcurve.add_patch(red_dot)

    plt.tight_layout()  # automatically adjust padding horizontally as well as vertically.
    return fig, ax_top, ax_eclipse, ax_lightcurve, red_dot


def ideal_velocity(sun, planet):
# Returns the velocity of the planet that is needed for a circular orbit around the sun in a 2 body system.
# https://de.wikipedia.org/wiki/Zweik%C3%B6rperproblem#Zeitparameter
    distance = np.sqrt(np.dot(sun.positions[0] - planet.positions[0], sun.positions[0] - planet.positions[0]))
    return np.sqrt(G * (sun.mass + planet.mass) / distance)


def ideal_radius(sun, planet, orbital_period=0):
# Returns the radius of the planet that is needed for a circular orbit around the sun in a 2 body system.
# If the orbital period is not given it is calculated from the planets velocity.
    mass = sun.mass + planet.mass
    if orbital_period > 0:
        return ((G * mass * orbital_period**2) / (4 * np.pi**2))**(1/3)
    else:
        planet_velocity = math.sqrt(np.dot(planet.velocity, planet.velocity))
        return G * mass / planet_velocity**2


def distance_2d_ecl(body1, body2, i):
    dx = body1.positions[i][0] - body2.positions[i][0]
    dz = body1.positions[i][2] - body2.positions[i][2]
    return math.sqrt((dx**2 + dz**2))


def distance_2d_top(body1, body2, i):
    dx = body1.positions[i][0] - body2.positions[i][0]
    dy = body1.positions[i][1] - body2.positions[i][1]
    return math.sqrt((dx**2 + dy**2))


def total_luminosity(bodies, stars, frame):
# add luminosity of all stars in the system while checking for eclipses
    luminosity = 0.0
    for star in stars:
        luminosity += star.luminosity
        for body in bodies:
            if body != star:
                eclipsed_area = star.eclipsed_by(body, frame)
                luminosity -= star.brightness * eclipsed_area
                # print(f'Eclipse {star.eclipsed_by(body):6.3e}')
    return luminosity


def calc_position_eclipse_luminosity(bodies, lightcurve):
    stars = [body for body in bodies if body.brightness > 0.0]
    lightcurve[0] = total_luminosity(bodies, stars, 0)
    for i in range(1, Iterations):
        clock = DT * i  # [s] time since start of animation
        for body1 in bodies:
            force = np.array([0.0, 0.0, 0.0])
            for body2 in bodies:
                if body1 != body2:
                    # Calculate distances between objects
                    distance_xyz = body2.positions[i-1] - body1.positions[i-1]
                    distance = np.sqrt(np.dot(distance_xyz, distance_xyz))
                    force_total = G * body1.mass * body2.mass / distance ** 2  # Use law of gravitation to calculate force acting on object
                    # Compute the force of attraction in each direction
                    x, y, z = distance_xyz[0], distance_xyz[1], distance_xyz[2]
                    polar_angle = math.acos(z / distance)
                    azimuth_angle = math.atan2(y, x)
                    force[0] += math.sin(polar_angle) * math.cos(azimuth_angle) * force_total
                    force[1] += math.sin(polar_angle) * math.sin(azimuth_angle) * force_total
                    force[2] += math.cos(polar_angle) * force_total
            acceleration = force / body1.mass  # Compute the acceleration in each direction
            body1.velocity += acceleration * DT  # Compute the velocity in each direction
            # Update positions
            movement = body1.velocity * DT - 0.5 * acceleration * DT ** 2
            body1.positions[i] = body1.positions[i-1] + movement
        lightcurve[i] = total_luminosity(bodies, stars, i)
        if i==1 or i % int(round(Iterations / 10)) == 0: # i > 0 and
            # print(f'{round(i / Iterations * 100):3d}% ', end="")
            print(f'\n{i:12d}: {distance_2d_top(bodies[0], bodies[1], i):.4e} ', end="")
    return 0


def update(frame, bodies, lightcurve, red_dot):
# first parameter comes from iterator frames (a parameter of FuncAnimation)
# the other parameters are given to this function via the parameter fargs of FuncAnimation.
    # Update patches. Send new circle positions to animation function.
    for body in bodies:  # top view: projection (x,y,z) -> (x,y), order = z
        body.circle_top.set(zorder=body.positions[frame * Sampling_rate][2])
        body.circle_top.center = body.positions[frame * Sampling_rate][0] / Scale_top, body.positions[frame * Sampling_rate][1] / Scale_top
    for body in bodies:  # eclipse view: projection (x,y,z) -> (x,z), order = -y
        body.circle_ecl.set(zorder=-body.positions[frame * Sampling_rate][1])
        body.circle_ecl.center = body.positions[frame * Sampling_rate][0] / Scale_ecl, body.positions[frame * Sampling_rate][2] / Scale_ecl
    red_dot.center = DT * Sampling_rate * frame, lightcurve[frame * Sampling_rate]
    if frame > 0 and frame % int(round(Frames/10)) == 0:
        print(f'{round(frame/Frames*100):3d}% ',end="")


### Main ####
# todo()
print(f'Producing {Frames/FPS:.0f} seconds long video, covering {DT*Iterations/60/60/24:5.2f} earth days. ({DT*Sampling_rate*FPS/60/60/24:.2f} earth days per video second.)')
lightcurve = np.zeros((Iterations))
bodies = []
init_bodies(bodies)

# print(ideal_velocity(bodies[0], bodies[1]))
# print(ideal_radius(bodies[0], bodies[1], orbital_period=1.1047*24*3600))
# exit(1234)

print(f'Calculating {Iterations:6d} iterations: ', end="")
tic = time.perf_counter()
calc_position_eclipse_luminosity(bodies, lightcurve)
toc = time.perf_counter()
print(f' 100%   {toc-tic:7.2f} seconds  ({Iterations/(toc-tic):.0f} iterations/second)')

# exit(12345)

sampled_lightcurve = np.take(lightcurve, range(0, Iterations, Sampling_rate))
fig, ax_top, ax_eclipse, ax_lightcurve, red_dot = init_plot(sampled_lightcurve) # adjust constants inside this function to fit your screen
for body in bodies:
    body.circle_top.set_color(body.color)
    body.circle_ecl.set_color(body.color)
    ax_top.add_patch(body.circle_top)
    ax_eclipse.add_patch(body.circle_ecl)
# Rendering
print(f'Animating {Frames:8d} frames:     ', end="")
tic = time.perf_counter()
anim = matplotlib.animation.FuncAnimation(fig, update, fargs=(bodies, lightcurve, red_dot), interval=1000 / FPS, frames=Frames, blit=False)
anim.save(video_file, fps=FPS, metadata={"title": " "}, extra_args=['-vcodec', 'libx264'])
toc = time.perf_counter()
print(f' 100%   {toc-tic:7.2f} seconds  ({Frames/(toc-tic):.0f} frames/second)')
print(f'{video_file} saved.')
# https://www.ffmpeg.org/libavcodec.html
# The libavcodec library provides a generic encoding/decoding framework and contains multiple decoders and encoders for audio, video, subtitle streams and bitstream filters.
# libx264 is a library for encoding mp4 videos.
