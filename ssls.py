# -*- coding: utf-8 -*-
# SSLS - Star System Lightcurve Simulator
# Body motion calculations inspired by https://colab.research.google.com/drive/1YKjSs8_giaZVrUKDhWLnUAfebuLTC-A5
# Needs ffmpeg to convert the data into a video. Download ffmpeg from https://www.ffmpeg.org/download.html. Extract the zip file and add "<yourdriveandpath>\FFmpeg\bin" to Environment Variable PATH.

def todo():
    0
    # Parameter für Lightcurve Zeit-Einheit einrichten.
    # Einheiten (Watt, Sekunden) an die Lightcurve schreiben?
    # Einheiten (AU, Meter) an eclipse- und top-view schreiben?
    # Lightcurve Zeit-Einheit automatisch oder per Parameter in Stunden/Tage/Jahre ändern? Dictionary vielleicht so?:
    TimeUnit = "hours"
    time_unit = {"seconds": ('s', 1), "hours": ('h', 3600), "years": ('h', 3600 * 24 * 365.25)}
    print("Test:", time_unit[TimeUnit])
    # Lightcurve y-Achse Einheit SL oder Watt?
    #
    # Automatisch *guten* Start-Geschwindigkeitsvektor berechnen.
    #
    # Lernen, wie man Bahndaten aus dem Internet zieht. Dann Venus ergänzen.
    #
    # Sonnenflecken. Keine Gravitation, aber Positionsupdate. Position leitet sich ab aus:
    # Position+Radius der Sonne, Rotationsgeschwindigkeit, Rotationsachse der Sonne; Längengrad, Breitengrad des Sonnenflecks.
    #
    # Parameter für Stern, der einen Faktor angibt, um den das Sternzentrum heller ist als der Sternrand. Noch besser: radiale Funktion statt Faktor?
    # Dann kann ich bei Eclipse die Helligkeit an der verdeckten Sternstelle interpolieren.

    # Funktion, die die Transitzeitpunkte ermittelt und speichert und Umlaufzeit-Variationen aufgrund zusätzlicher Körper erfasst.



import configparser
import math
import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
import time


class Body():
    def __init__(self, name="no_name", mass=1.0, radius=1.0, luminosity=0.0, startposition=np.array([0.0, 0.0, 0.0]), velocity=np.array([0.0, 0.0, 0.0]), color=(0.0, 0.0, 0.0)):
        self.name = name                                # name
        self.mass = mass                                # [kg]
        self.radius = radius                            # [m]
        self.area_2d = np.pi * radius**2                # [m**2]
        self.luminosity = luminosity                    # [W]
        self.brightness = luminosity / self.area_2d     # luminosity per (apparent) area [W/m**2]
        self.positions = np.zeros((Iterations,3))       # position for each frame
        self.positions[0] = startposition               # [m] initial position
        self.velocity = velocity                        # [m/s] (np.array)
        self.color = color                              # (R, G, B)  each between 0 and 1
        if luminosity > 0:
            extrascale_ecl, extrascale_top = StarScale_ecl, StarScale_top  # It's a star. Scale radius in plot accordingly.
        else:
            extrascale_ecl, extrascale_top = PlanetScale_ecl, PlanetScale_top # It's a planet. Scale radius in plot accordingly.
        self.circle_top = matplotlib.patches.Circle((0, 0), radius * extrascale_top / Scale_top)  # matplotlib patch for top view
        self.circle_ecl = matplotlib.patches.Circle((0, 0), radius * extrascale_ecl / Scale_ecl)  # matplotlib patch for eclipsed view


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


def init_bodies(bodies, config):
    for section in config.sections():
        if section not in ["Astronomical Constants", "Video", "Plot", "Scale"]:
            bodies.append(Body(name=section,
                               mass=eval(config.get(section, "mass")),
                               radius=eval(config.get(section, "radius")),
                               startposition=np.array([eval(x) for x in config.get(section, "startposition").split(",")], dtype=float),
                               velocity=np.array([eval(x) for x in config.get(section, "velocity").split(",")], dtype=float),
                               luminosity=eval(config.get(section, "luminosity")),
                               color=tuple([eval(x) for x in config.get(section, "color").split(",")])))


def init_plot(sampled_lightcurve):
    fig = plt.figure()
    fig.set_figwidth(FigureWidth)
    fig.set_figheight(FigureHeight)
    fig.set_facecolor("black")  # background color outside of ax_eclipse and ax_lightcurve
    buffer = 0.0
    fig.subplots_adjust(left=buffer, right=1.0-buffer, bottom=buffer, top=1-buffer)  # Positions of the subplots edges, as a fraction of the figure width.

    ax_eclipse = plt.subplot2grid(shape=(5, 2), loc=(0, 0), rowspan=4, colspan=1)
    ax_eclipse.set_xlim(-Xlim, Xlim)
    ax_eclipse.set_ylim(-Ylim, Ylim)
    ax_eclipse.set_aspect('equal')
    ax_eclipse.set_facecolor("black")  # background color
    # ax_eclipse.get_xaxis().set_visible(False)
    # ax_eclipse.get_yaxis().set_visible(False)

    ax_top = plt.subplot2grid(shape=(5, 2), loc=(0, 1), rowspan=4, colspan=1)
    ax_top.set_xlim(-Xlim, Xlim)
    ax_top.set_ylim(-Ylim, Ylim)
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
    ax_lightcurve.set_xlabel("time [s]")
    ax_lightcurve.set_ylabel("luminosity [W]")
    ax_lightcurve.plot(range(0, round(Iterations * DT), round(Sampling_rate * DT)), sampled_lightcurve, color="white")
    red_dot = matplotlib.patches.Ellipse((0,0), Iterations * DT * RedDotWidth, scope * RedDotHeight)  # matplotlib patch
    red_dot.set(zorder=2)  # Dot in front of lightcurve
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
        if i % int(round(Iterations / 10)) == 0:
            print(f'{round(i / Iterations * 100):3d}% ', end="")
            # print(f'\n{i:12d}: {distance_2d_top(bodies[0], bodies[1], i):.4e} ', end="")
    return 0


def update(frame, bodies, lightcurve, red_dot):
# Update patches. Send new circle positions to animation function.
# First parameter comes from iterator frames (a parameter of FuncAnimation).
# The other parameters are given to this function via the parameter fargs of FuncAnimation.
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


# read program parameters #
config = configparser.ConfigParser(inline_comment_prefixes='#')#, interpolation=configparser.ExtendedInterpolation())
config.read('ssls.ini')
# [Astronomical Constants]
G  = eval(config.get("Astronomical Constants", "G"))
AU = eval(config.get("Astronomical Constants", "AU"))
SR = eval(config.get("Astronomical Constants", "SR"))
SM = eval(config.get("Astronomical Constants", "SM"))
SL = eval(config.get("Astronomical Constants", "SL"))
ER = eval(config.get("Astronomical Constants", "ER"))
EM = eval(config.get("Astronomical Constants", "EM"))
EV = eval(config.get("Astronomical Constants", "EV"))
# [Video]
Video_file    = config.get("Video", "Video_file")
Frames        = eval(config.get("Video", "Frames"))
FPS           = eval(config.get("Video", "FPS"))
DT            = eval(config.get("Video", "DT"))
Sampling_rate = eval(config.get("Video", "Sampling_rate"))
Iterations    = Frames * Sampling_rate
# [Plot]
FigureWidth   = eval(config.get("Plot", "FigureWidth"))
FigureHeight  = eval(config.get("Plot", "FigureHeight"))
Xlim          = eval(config.get("Plot", "Xlim"))
Ylim          = eval(config.get("Plot", "Ylim"))
RedDotHeight  = eval(config.get("Plot", "RedDotHeight"))
RedDotWidth   = eval(config.get("Plot", "RedDotWidth"))
# [Scale]
Scale_ecl       = eval(config.get("Scale", "Scale_ecl"))
StarScale_ecl   = eval(config.get("Scale", "StarScale_ecl"))
PlanetScale_ecl = eval(config.get("Scale", "PlanetScale_ecl"))
Scale_top       = eval(config.get("Scale", "Scale_top"))
StarScale_top   = eval(config.get("Scale", "StarScale_top"))
PlanetScale_top = eval(config.get("Scale", "PlanetScale_top"))


# Init
print(f'Producing {Frames/FPS:.0f} seconds long video, covering {DT*Iterations/60/60/24:5.2f} earth days. ({DT*Sampling_rate*FPS/60/60/24:.2f} earth days per video second.)')
lightcurve = np.zeros((Iterations))
bodies = []
init_bodies(bodies, config)
# exit(888)

# print(ideal_velocity(bodies[0], bodies[1]))
# print(ideal_radius(bodies[0], bodies[1], orbital_period=1.1047*24*3600))
# exit(1234)

# Calculate body positions and the resulting lightcurve
print(f'Calculating {Iterations:6d} iterations: ', end="")
tic = time.perf_counter()
calc_position_eclipse_luminosity(bodies, lightcurve)
toc = time.perf_counter()
print(f' 100%   {toc-tic:7.2f} seconds  ({Iterations/(toc-tic):.0f} iterations/second)')

# Prepare animation
sampled_lightcurve = np.take(lightcurve, range(0, Iterations, Sampling_rate))  # use only some of the calculated positions for the animation because it is so slow
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
anim.save(Video_file, fps=FPS, metadata={"title": " "}, extra_args=['-vcodec', 'libx264'])  # https://www.ffmpeg.org/libavcodec.html
toc = time.perf_counter()
print(f' 100%   {toc-tic:7.2f} seconds  ({Frames/(toc-tic):.0f} frames/second)')
print(f'{Video_file} saved.')
