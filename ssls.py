# -*- coding: utf-8 -*-
# SSLS - Star System Lightcurve Simulator
# The SSLS calculates the movements and eclipses of celestial bodies and produces a video of this.
# Specify mass, radius and other properties of some stars and planets in a configuration file. Then run "ssls.py <configfilename>" to produce the video.
# The video shows simultanously a view of the star system from top and from the side and the lightcurve of the system's total luminosity over time.
# Usually you do not need to look at or even modify the python code. Instead control the program's outcome with the config file.
# The meaning of all program parameters is documented in the config file.
# SSLS uses ffmpeg to convert the data into a video. Download ffmpeg from https://www.ffmpeg.org/download.html. Extract the zip file and add "<yourdriveandpath>\FFmpeg\bin" to Environment Variable PATH.
# Your questions and comments are welcome. Just open an issue on https://github.com/lichtgestalter/ssls/issues to get my attention :)



import configparser
import math
import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
import sys
import time


class Parameters:
    def __init__(self):
        """Read program parameters and properties of the physical bodies from config file."""
        standard_sections = ["Astronomical Constants", "Video", "Plot", "Scale"]
        config = configparser.ConfigParser(inline_comment_prefixes='#')
        if len(config.read(find_config_file())) < 1:
            raise Exception("Config file not found. Check program parameter.")
        for section in standard_sections:
            if section not in config.sections():
                raise Exception(f'Section {section} missing in config file.')

        # [Astronomical Constants]
        g = eval(config.get("Astronomical Constants", "g"))  # For ease of use of these units in the config file they are additionally defined here without the prefix "self.".
        au = eval(config.get("Astronomical Constants", "au"))
        r_sun = eval(config.get("Astronomical Constants", "r_sun"))
        m_sun = eval(config.get("Astronomical Constants", "m_sun"))
        l_sun = eval(config.get("Astronomical Constants", "l_sun"))
        r_earth = eval(config.get("Astronomical Constants", "r_earth"))
        m_earth = eval(config.get("Astronomical Constants", "m_earth"))
        v_earth = eval(config.get("Astronomical Constants", "v_earth"))
        self.g, self.au, self.r_sun, self.m_sun, self.l_sun, self.r_earth, self.m_earth, self.v_earth = g, au, r_sun, m_sun, l_sun, r_earth, m_earth, v_earth

        # [Video]
        self.video_file = config.get("Video", "video_file")
        self.frames = eval(config.get("Video", "frames"))
        self.fps = eval(config.get("Video", "fps"))
        self.dt = eval(config.get("Video", "dt"))
        self.sampling_rate = eval(config.get("Video", "sampling_rate"))
        self.iterations = self.frames * self.sampling_rate

        # [Scale]
        self.scope_ecl = eval(config.get("Scale", "scope_ecl"))
        self.star_scale_ecl = eval(config.get("Scale", "star_scale_ecl"))
        self.planet_scale_ecl = eval(config.get("Scale", "planet_scale_ecl"))
        self.scope_top = eval(config.get("Scale", "scope_top"))
        self.star_scale_top = eval(config.get("Scale", "star_scale_top"))
        self.planet_scale_top = eval(config.get("Scale", "planet_scale_top"))

        # [Plot]
        self.figure_width = eval(config.get("Plot", "figure_width"))
        self.figure_height = eval(config.get("Plot", "figure_height"))
        self.xlim = eval(config.get("Plot", "xlim"))
        self.ylim = eval(config.get("Plot", "ylim"))
        self.time_units = {"s": 1, "min": 60, "h": 3600, "d": 24 * 3600, "mon": 365.25 * 24 * 3600 / 12, "y": 365.25 * 24 * 3600}
        self.x_unit_name = config.get("Plot", "x_unit")
        self.x_unit_value = self.time_units[self.x_unit_name]
        self.red_dot_height = eval(config.get("Plot", "red_dot_height"))
        self.red_dot_width = eval(config.get("Plot", "red_dot_width"))

        # Checking all parameters defined so far
        for key in vars(self):
            if type(getattr(self, key)) not in [str, dict]:
                if getattr(self, key) <= 0:
                    raise Exception(f"No parameter in sections {standard_sections} may be zero or negative.")

        # Physical bodies
        for section in config.sections():
            if section not in standard_sections:
                bodies.append(Body(p=self,
                                   name=section,
                                   mass=eval(config.get(section, "mass")),
                                   radius=eval(config.get(section, "radius")),
                                   luminosity=eval(config.get(section, "luminosity")),
                                   startposition=np.array([eval(x) for x in config.get(section, "startposition").split(",")], dtype=float),
                                   velocity=np.array([eval(x) for x in config.get(section, "velocity").split(",")], dtype=float),
                                   beta=eval(config.get(section, "beta")),
                                   color=tuple([eval(x) for x in config.get(section, "color").split(",")])))

        # Checking parameters of physical bodies
        if len(bodies) < 1:
            raise Exception("No physical bodies specified.")
        for body in bodies:
            if body.radius <= 0:
                raise Exception(f'{body.name} has invalid radius {body.radius}.')
            if body.mass <= 0:
                raise Exception(f'{body.name} has invalid mass {body.mass}.')
            if body.luminosity <= 0:
                raise Exception(f'{body.name} has invalid luminosity {body.luminosity}.')
            if body.beta <= 0:
                raise Exception(f'{body.name} has invalid limb darkening parameter beta {body.beta}.')
            for c in body.color:
                if c < 0 or c > 1:
                    raise Exception(f'{body.name} has invalid color value {c}.')


class Body:
    def __init__(self, p, name, mass, radius, luminosity, startposition, velocity, beta, color=(.5, .5, .5)):
        """Initialize instance of physical body."""
        self.name = name                                # name
        self.mass = mass                                # [kg]
        self.radius = radius                            # [m]
        self.area_2d = math.pi * radius**2                # [m**2]
        self.luminosity = luminosity                    # [W]
        self.brightness = luminosity / self.area_2d     # luminosity per (apparent) area [W/m**2]
        self.positions = np.zeros((p.iterations, 3), dtype=float)  # position for each frame
        self.positions[0] = startposition               # [m] initial position
        self.velocity = velocity                        # [m/s] (np.array)
        self.color = color                              # (R, G, B)  each between 0 and 1

        if luminosity > 0:
            extrascale_ecl, extrascale_top = p.star_scale_ecl, p.star_scale_top  # It's a star. Scale radius in plot accordingly.
            self.beta = beta                            # [1]
        else:
            extrascale_ecl, extrascale_top = p.planet_scale_ecl, p.planet_scale_top  # It's a planet. Scale radius in plot accordingly.
            self.beta = 0.0
        self.circle_top = matplotlib.patches.Circle((0, 0), radius * extrascale_top / p.scope_top)  # Matplotlib patch for top view
        self.circle_ecl = matplotlib.patches.Circle((0, 0), radius * extrascale_ecl / p.scope_ecl)  # Matplotlib patch for eclipsed view

        self.d, self.h, self.angle, self.eclipsed_area = 0.0, 0.0, 0.0, 0.0  # Used for calculation of eclipsed area in function eclipsed_by.

    def eclipsed_by(self, body, iteration):
        """Returns area, relative_radius
        area: Area of self which is eclipsed by body.
        relative_radius: The distance of the approximated center of the eclipsed area from the center of self as a percentage of self.radius (used for limb darkening)."""
        if body.positions[iteration][1] < self.positions[iteration][1]:  # Is body nearer to viewpoint than self? (i.e. its position has a smaller y-coordinate)
            d = distance_2d_ecl(body, self, iteration)
            if d < self.radius + body.radius:  # Does body eclipse self?
                if d <= abs(self.radius - body.radius):  # Annular (i.e. ring) eclipse or total eclipse
                    if self.radius < body.radius:  # Total eclipse
                        area = self.area_2d
                        relative_radius = 0
                        # print(f'  total: {iteration:7d}  rel.area: {area/self.area_2d*100:6.0f}%  rel.r: {relative_radius*100:6.0f}%')
                        return area, relative_radius
                    else:  # Annular (i.e. ring) eclipse
                        area = body.area_2d
                        relative_radius = d / self.radius
                        # print(f'   ring: {iteration:7d}  rel.area: {area / self.area_2d * 100:6.0f}%  rel.r: {relative_radius * 100:6.0f}%')
                        return area, relative_radius
                else:  # Partial eclipse
                    # Eclipsed area is the sum of a circle segment of self plus a circle segment of body
                    # https://de.wikipedia.org/wiki/Kreissegment  https://de.wikipedia.org/wiki/Schnittpunkt#Schnittpunkte_zweier_Kreise
                    self.d = (self.radius**2 - body.radius**2 + d**2) / (2 * d)  # Distance of center from self to radical axis
                    body.d = (body.radius**2 - self.radius**2 + d**2) / (2 * d)  # Distance of center from body to radical axis
                    body.h = body.radius + self.d - d  # Height of circle segment
                    self.h = self.radius + body.d - d  # Height of circle segment
                    body.angle = 2 * math.acos(1 - body.h / body.radius)  # Angle of circle segment
                    self.angle = 2 * math.acos(1 - self.h / self.radius)  # Angle of circle segment
                    body.eclipsed_area = body.radius**2 * (body.angle - math.sin(body.angle)) / 2  # Area of circle segment
                    self.eclipsed_area = self.radius**2 * (self.angle - math.sin(self.angle)) / 2  # Area of circle segment
                    area = body.eclipsed_area + self.eclipsed_area  # Eclipsed area is sum of two circle segments.
                    relative_radius = (self.radius + self.d - body.h) / (2 * self.radius)  # Relative distance between approximated center of eclipsed area and center of self
                    # print(f'partial: {iteration:7d}  rel.area: {area/self.area_2d*100:6.0f}%  rel.r: {relative_radius*100:6.0f}%')
                    return area, relative_radius
            else:  # No eclipse because, seen from viewer, the bodies are not close enough to each other
                return 0.0, 0.0
        else:  # body cannot eclipse self, because self is nearer to viewer than body
            return 0.0, 0.0


def find_config_file():
    """Check program parameters and extract config file name from them."""
    if len(sys.argv) == 1:
        print("Using default config file ssls.ini. Specify config file name as program parameter if you want to use another config file.")
        return 'ssls.ini'
    elif len(sys.argv) == 2:
        return sys.argv[1]
    else:
        print("First program parameter is interpreted as config file name. Further parameters are ignored.")
        return sys.argv[1]


def ideal_velocity(sun, planet):
    """Utility function. Not used in main program.
    Returns the velocity of the planet that is needed for a circular orbit around the sun in a 2 body system.
    https://de.wikipedia.org/wiki/Zweik%C3%B6rperproblem#Zeitparameter"""
    distance = math.sqrt(np.dot(sun.positions[0] - planet.positions[0], sun.positions[0] - planet.positions[0]))
    return math.sqrt(P.g * (sun.mass + planet.mass) / distance)


def ideal_radius(sun, planet, orbital_period=0):
    """Utility function. Not used in main program.
    Returns the radius of the planet that is needed for a circular orbit around the sun in a 2 body system.
    If the orbital period is not given it is calculated from the planets velocity."""
    mass = sun.mass + planet.mass
    if orbital_period > 0:
        return ((P.g * mass * orbital_period**2) / (4 * math.pi**2))**(1/3)
    else:
        planet_velocity = math.sqrt(np.dot(planet.velocity, planet.velocity))
        return P.g * mass / planet_velocity**2


def distance_2d_ecl(body1, body2, i):
    """Return distance of the centers of 2 physical bodies as seen by a viewer (projection y->0)."""
    dx = body1.positions[i][0] - body2.positions[i][0]
    dz = body1.positions[i][2] - body2.positions[i][2]
    return math.sqrt((dx**2 + dz**2))


def limbdarkening(relative_radius, beta):
    """https://en.wikipedia.org/wiki/Limb_darkening
    https://de.wikipedia.org/wiki/Photosph%C3%A4re#Mitte-Rand-Verdunkelung
    Approximates the flux of a star at a point on the star seen from a very large distance.
    The point's apparent distance from the star's center is relative_radius * radius.
    Beta depends on the wavelength. 2.3 is a good compromise for the spectrum of visible light."""
    if relative_radius >= 1:
        return 1 / (1 + beta)
    return (1 + beta * math.sqrt(1 - relative_radius**2)) / (1 + beta)


def total_luminosity(stars, iteration):
    """"Add luminosity of all stars in the system while checking for eclipses
    does not yet work correctly for eclipsed eclipses (three or more bodies in line of sight at the same time)."""
    luminosity = 0.0
    for star in stars:
        luminosity += star.luminosity
        for body in bodies:
            if body != star:
                eclipsed_area, relative_radius = star.eclipsed_by(body, iteration)
                if eclipsed_area != 0:
                    luminosity -= star.brightness * eclipsed_area * limbdarkening(relative_radius, star.beta)
    return luminosity


def calc_positions_eclipses_luminosity():
    """Calculate distances, forces, accelerations, velocities of the bodies for each iteration.
    The resulting body positions and the lightcurve are stored for later use in the animation.
    Body motion calculations inspired by https://colab.research.google.com/drive/1YKjSs8_giaZVrUKDhWLnUAfebuLTC-A5."""
    stars = [body for body in bodies if body.brightness > 0.0]
    lightcurve[0] = total_luminosity(stars, 0)
    for iteration in range(1, P.iterations):
        for body1 in bodies:
            force = np.array([0.0, 0.0, 0.0])
            for body2 in bodies:
                if body1 != body2:
                    # Calculate distances between bodies:
                    distance_xyz = body2.positions[iteration-1] - body1.positions[iteration-1]
                    distance = math.sqrt(np.dot(distance_xyz, distance_xyz))
                    force_total = P.g * body1.mass * body2.mass / distance ** 2  # Use law of gravitation to calculate force acting on body.
                    # Compute the force of attraction in each direction:
                    x, y, z = distance_xyz[0], distance_xyz[1], distance_xyz[2]
                    polar_angle = math.acos(z / distance)
                    azimuth_angle = math.atan2(y, x)
                    force[0] += math.sin(polar_angle) * math.cos(azimuth_angle) * force_total
                    force[1] += math.sin(polar_angle) * math.sin(azimuth_angle) * force_total
                    force[2] += math.cos(polar_angle) * force_total
            acceleration = force / body1.mass  # Compute the acceleration in each direction.
            body1.velocity += acceleration * P.dt  # Compute the velocity in each direction.
            # Update positions:
            movement = body1.velocity * P.dt - 0.5 * acceleration * P.dt ** 2
            body1.positions[iteration] = body1.positions[iteration-1] + movement
        lightcurve[iteration] = total_luminosity(stars, iteration)  # Update lightcurve.
        if iteration % int(round(P.iterations / 10)) == 0:  # Inform user about program's progress.
            print(f'{round(iteration / P.iterations * 100):3d}% ', end="")
    return 0


def calc_physics():
    """Calculate body positions and the resulting lightcurve."""
    print(f'Producing {P.frames / P.fps:.0f} seconds long video, covering {P.dt * P.iterations / 60 / 60 / 24:5.2f} earth days. ({P.dt * P.sampling_rate * P.fps / 60 / 60 / 24:.2f} earth days per video second.)')
    print(f'Calculating {P.iterations:6d} iterations: ', end="")
    tic = time.perf_counter()
    calc_positions_eclipses_luminosity()
    toc = time.perf_counter()
    print(f' 100%   {toc-tic:7.2f} seconds  ({P.iterations / (toc - tic):.0f} iterations/second)')


def tic_delta(scope):
    """Returns a distance between two tics on an axis so that the total number of tics on that axis is between 5 and 10."""
    delta = 10 ** np.floor(math.log10(scope))
    if scope/delta < 5:
        if scope/delta < 2:
            return delta/5
        else:
            return delta/2
    else:
        return delta


def init_plot(sampled_lightcurve):
    """Initialize the matplotlib figure containing 3 axis:
    Eclipse view (top left): projection (x,y,z) -> (x,z), order = -y.
    Top view (top right): projection (x,y,z) -> (x,y), order = z.
    Lightcurve (bottom)"""
    fig = plt.figure()
    fig.set_figwidth(P.figure_width)
    fig.set_figheight(P.figure_height)
    fig.set_facecolor("black")  # background color outside of ax_eclipse and ax_lightcurve
    buffer = 0
    fig.subplots_adjust(left=buffer, right=1.0-buffer, bottom=buffer, top=1-buffer)  # Positions of the subplots edges, as a fraction of the figure width.

    ax_eclipse = plt.subplot2grid(shape=(5, 2), loc=(0, 0), rowspan=4, colspan=1)
    ax_eclipse.set_xlim(-P.xlim, P.xlim)
    ax_eclipse.set_ylim(-P.ylim, P.ylim)
    ax_eclipse.set_aspect('equal')
    ax_eclipse.set_facecolor("black")  # background color
    # ax_eclipse.get_xaxis().set_visible(False)
    # ax_eclipse.get_yaxis().set_visible(False)

    ax_top = plt.subplot2grid(shape=(5, 2), loc=(0, 1), rowspan=4, colspan=1)
    ax_top.set_xlim(-P.xlim, P.xlim)
    ax_top.set_ylim(-P.ylim, P.ylim)
    ax_top.set_aspect('equal')
    ax_top.set_facecolor("black")  # background color

    ax_lightcurve = plt.subplot2grid(shape=(5, 1), loc=(4, 0), rowspan=1, colspan=1)
    ax_lightcurve.set_facecolor("black")  # background color

    ax_lightcurve.tick_params(axis='x', colors='grey')
    xmax = P.iterations * P.dt / P.x_unit_value
    ax_lightcurve.set_xlim(0, xmax)
    xvalues = [x * tic_delta(xmax) for x in range(round(xmax / tic_delta(xmax)))]
    xlabels = [f'{round(x,4)} {P.x_unit_name}' for x in xvalues]
    ax_lightcurve.set_xticks(xvalues, labels=xlabels)

    ax_lightcurve.tick_params(axis='y', colors='grey')
    minl = lightcurve.min(initial=None)
    maxl = lightcurve.max(initial=None)
    scope = maxl - minl
    buffer = 0.05 * scope
    ax_lightcurve.set_ylim(minl - buffer, maxl + buffer)

    ticdelta = tic_delta(maxl-minl)
    yvalues = [1 - y * ticdelta for y in range(round(float((maxl-minl) / ticdelta)))]
    ylabels = [f'{round(100*y,10)} %' for y in yvalues]
    ax_lightcurve.set_yticks(yvalues, labels=ylabels)

    time_axis = np.arange(0, round(P.iterations * P.dt), round(P.sampling_rate * P.dt), dtype=float)
    time_axis /= P.x_unit_value
    ax_lightcurve.plot(time_axis, sampled_lightcurve[0:len(time_axis)], color="white")

    red_dot = matplotlib.patches.Ellipse((0, 0), P.iterations * P.dt * P.red_dot_width / P.x_unit_value, scope * P.red_dot_height)  # matplotlib patch
    red_dot.set(zorder=2)  # Dot in front of lightcurve.
    red_dot.set_color((1, 0, 0))  # red
    ax_lightcurve.add_patch(red_dot)
    plt.tight_layout()  # Automatically adjust padding horizontally as well as vertically.
    return fig, ax_top, ax_eclipse, ax_lightcurve, red_dot


def prepare_animation():
    """Initialize all matplotlib objects."""
    sampled_lightcurve = np.take(lightcurve, range(0, P.iterations, P.sampling_rate))  # Use only some of the calculated positions for the animation because it is so slow.
    fig, ax_top, ax_eclipse, ax_lightcurve, red_dot = init_plot(sampled_lightcurve)  # Adjust constants in section [Plot] of config file to fit your screen.
    for body in bodies:  # Circles represent the bodies in the animation. Set their colors and add them to the matplotlib axis.
        body.circle_top.set_color(body.color)
        body.circle_ecl.set_color(body.color)
        ax_top.add_patch(body.circle_top)
        ax_eclipse.add_patch(body.circle_ecl)
    return fig, red_dot


def next_animation_frame(frame, red_dot):
    """Update patches. Send new circle positions to animation function.
    First parameter comes from iterator frames (a parameter of FuncAnimation).
    The other parameters are given to this function via the parameter fargs of FuncAnimation."""
    for body in bodies:  # Top view: projection (x,y,z) -> (x,y), order = z
        body.circle_top.set(zorder=body.positions[frame * P.sampling_rate][2])
        body.circle_top.center = body.positions[frame * P.sampling_rate][0] / P.scope_top, body.positions[frame * P.sampling_rate][1] / P.scope_top
    for body in bodies:  # Eclipse view: projection (x,y,z) -> (x,z), order = -y
        body.circle_ecl.set(zorder=-body.positions[frame * P.sampling_rate][1])
        body.circle_ecl.center = body.positions[frame * P.sampling_rate][0] / P.scope_ecl, body.positions[frame * P.sampling_rate][2] / P.scope_ecl
    red_dot.center = P.dt * P.sampling_rate * frame / P.x_unit_value, lightcurve[frame * P.sampling_rate]
    if frame > 0 and frame % int(round(P.frames / 10)) == 0:  # Inform user about program's progress.
        print(f'{round(frame / P.frames * 100):3d}% ', end="")


def render_animation(fig, red_dot):
    """Calls next_animation_frame() for each frame and saves the video."""
    print(f'Animating {P.frames:8d} frames:     ', end="")
    tic = time.perf_counter()
    anim = matplotlib.animation.FuncAnimation(fig, next_animation_frame, fargs=(red_dot,), interval=1000 / P.fps, frames=P.frames, blit=False)
    anim.save(P.video_file, fps=P.fps, metadata={"title": " "}, extra_args=['-vcodec', 'libx264'])  # https://www.ffmpeg.org/libavcodec.html
    toc = time.perf_counter()
    print(f' 100%   {toc-tic:7.2f} seconds  ({P.frames / (toc - tic):.0f} frames/second)')
    print(f'{P.video_file} saved.')


# main program
bodies = []  # Will contain all physical objects of the simulation.
P = Parameters()  # Read program parameters and properties of the physical bodies from config file.
lightcurve = np.zeros(P.iterations)  # Initialize lightcurve.
calc_physics()  # Calculate body positions and the resulting lightcurve.
lightcurve /= lightcurve.max(initial=None)  # Normalize flux.
render_animation(*prepare_animation())


# print(ideal_velocity(bodies[0], bodies[1]))
# print(ideal_radius(bodies[0], bodies[1], orbital_period=1.1047*P.time_units["d"]))
# exit(1234)
