import gdsfactory  as gf



def create_component():
    c = gf.Component()
    return c

def taper(length, width1, width2, crss_section):
    c = gf.components.taper(length=length, width1=width1, width2=width2, cross_section=crss_section)
    # el cross section es la forma
    # que adopta la seccion transversal, en este caso es cuadrado
    return c

def str_line(length, crss_section):
    # Import the structure
    c = gf.components.straight(length=length, cross_section=crss_section)
    return c

def snake_structure(spiral_length,num_spirals,spiral_style, width ):
    c = gf.components.delay_snake2(
        length=spiral_length,
        length0=0,
        length2=0,
        n=num_spirals,
        bend180=spiral_style,
        cross_section=gf.cross_section.strip(width = width)
    )

    return c

def bend_s (bend_length_x,bend_length_y, crss_section):
    c = gf.components.bend_s(
            size=(bend_length_x, bend_length_y),
            cross_section = crss_section
        )
    return c

def coupler (gap, length, bend_length_x, bend_length_y, crss_section):

    bend_s_structure= bend_s(bend_length_x, bend_length_y, crss_section)

    c = gf.components.coupler(
                gap=gap,
                length=length,
                dy=bend_length_y,
                dx=bend_length_x,
                cross_section = crss_section,
                allow_min_radius_violation=False,
                bend = bend_s_structure
            )

    return c

def text_on_structure(text,size,pos, justify, layer):

    c = gf.components.text(
                text=text,
                size=size,
                position=(pos[0], pos[1]),
                justify=justify,
                layer=layer
            )

    return c

def spiral_with_taper_and_snake(n_values,
                                spiral_lengths,
                                width_up,
                                width_down,
                                y_offset,
                                initial_pos_x,
                                initial_pos_y,
                                chip_length=11000,
                                margin=200):
    """
    Generates multiple structures consisting of:
    - An input straight waveguide
    - An input taper
    - A snake (spiral) waveguide
    - An output taper
    - An output straight waveguide

    Each structure ensures proper spacing within the chip dimensions.

    Args:
        n_values (list): List of `n` values (number of bends in the snake).
        spiral_lengths (list): List of snake (spiral) lengths.
        y_offset (float): Vertical spacing between each generated structure.
        chip_length (float): Total chip length constraint.
        margin (float): Safety margin at the chip edges.
    """

    c = create_component()

    for idx, (n, spiral_length) in enumerate(zip(n_values, spiral_lengths)):
        structure = create_component()

        # Base parameters
        taper_length = 300  # Fixed taper length
        usable_length = chip_length - 2 * margin  # Available space on the chip

        # Generate the snake (spiral) component with the given parameters
        snake_component = snake_structure(spiral_length, n, "bend_euler180", width_down)

        snake_length_real = snake_component.size_info.width  # Actual X length of the snake

        # Compute the remaining space for the straight waveguides
        remaining_length = usable_length - (1000 + taper_length + snake_length_real + taper_length + 1000)

        if remaining_length < 0:
            raise ValueError(f"The total design length exceeds {chip_length} µm.")

        straight_length = remaining_length / 2  # Divide evenly for both straight waveguides

        # 1. Create the first straight waveguide (input)
        straight_waveguide1 = structure.add_ref(str_line(1000,gf.cross_section.strip(width=width_up)))
        structure.add_port(name=f"o{2*idx+1}", port=straight_waveguide1.ports["o1"])  # Define correct input port

        # 2. Create the first taper (3.0 µm → 0.8 µm)
        taper_in = structure.add_ref(taper(taper_length,width_up,width_down,gf.cross_section.strip)
        )
        taper_in.connect("o1", straight_waveguide1.ports["o2"])

        # 3. Create the second straight waveguide (before the snake)
        straight_waveguide2 = structure.add_ref(
            # gf.components.straight(length=straight_length, cross_section=gf.cross_section.strip(width=0.8))
            str_line(straight_length, gf.cross_section.strip(width=width_down))
        )
        straight_waveguide2.connect("o1", taper_in.ports["o2"])

        # 4. Add the snake (spiral) structure
        snake = structure.add_ref(snake_component)
        snake.connect("o1", straight_waveguide2.ports["o2"])

        # 5. Create the third straight waveguide (after the snake)
        straight_waveguide3 = structure.add_ref(str_line(straight_length, gf.cross_section.strip(width=width_down)))
        straight_waveguide3.connect("o1", snake.ports["o2"])

        # 6. Create the second taper (0.8 µm → 3.0 µm)
        taper_out = structure.add_ref(taper(taper_length, width_down, width_up, gf.cross_section.strip))
        taper_out.connect("o1", straight_waveguide3.ports["o2"])

        # 7. Create the last straight waveguide (output)
        straight_waveguide4 = structure.add_ref(str_line(1000, gf.cross_section.strip(width=width_up)))
        straight_waveguide4.connect("o1", taper_out.ports["o2"])

        # **Labels**
        structure.add_ref(text_on_structure(f"Spiral length: {spiral_length}, n-spirals: {n}", 50, (straight_waveguide1.xmin + 500, straight_waveguide1.ymax + 150),"left", "WG"))

        structure.add_ref(text_on_structure(f"Spiral length: {spiral_length}, n-spirals: {n}", 50, (straight_waveguide4.xmin - 500, straight_waveguide4.ymax + 150),"left", "WG"))

        # 8. Define the correct output port name
        structure.add_port(name=f"o{2*idx+2}", port=straight_waveguide4.ports["o2"])

        # Add the complete structure to `c` and position it with an offset in Y
        ref = c.add_ref(structure)
        ref.move(( initial_pos_x , initial_pos_y + idx * y_offset))  # Offset each structure vertically

        # Re-add the ports to the main component
        c.add_port(name=f"o{2*idx+1}", port=ref.ports[f"o{2*idx+1}"])
        c.add_port(name=f"o{2*idx+2}", port=ref.ports[f"o{2*idx+2}"])

    x_final_pos = 0
    y_final_pos = c.ymax

    return c,x_final_pos,y_final_pos


def multiple_directional_Couplers_50_to_50_with_SBend(num_of_couplers,
                                                      initial_pos_x,
                                                      initial_pos_y,
                                                      y_offset,
                                                      coupler_width,
                                                      gap,
                                                      inner_s_bend_x,
                                                      inner_s_bend_y,
                                                      outter_s_bend_x,
                                                      outter_s_bend_y,  # S-Bend (width, height)
                                                      laser_lambda,
                                                      n1,
                                                      n2,
                                                      chip_length=11000,  # Total allowed chip length
                                                      margin=200          # Safe margins
                                                      ):
    """
    Generates directional couplers with correctly oriented S-Bends at input and output.

    Returns:
        gf.Component: Multiple directional couplers with properly adjusted S-Bends.
    """

    if gap < 0.4:
        raise ValueError(f"Invalid gap: {gap} µm. Must be >= 0.4 µm.")

    c = gf.Component()
    # Example base_length (approximation)
    base_length = laser_lambda / ( 4* (n1 - n2)) # L = lamba/(2 * (delta n)) para 780 nm y

    for idx in range(1, num_of_couplers+1):
        structure = gf.Component()
        length = base_length

        total_fixed_length = (
                2 * (1000 + 300)  # waveguides + tapers
                + length
                + 2 * outter_s_bend_x
        )

        usable_length = chip_length - 2 * margin
        remaining_length = max((usable_length - total_fixed_length) / 2, 100)

        coupler_structure = structure.add_ref(coupler(gap,length,inner_s_bend_x,inner_s_bend_y,gf.cross_section.strip(width=coupler_width)))

        #  **Lower Input Path**
        straight1_down = structure.add_ref( str_line(1000, gf.cross_section.strip(width=3.0)) )

        taper1_down = structure.add_ref(taper(300,3.0,coupler_width,"strip"))

        s_bend1_down = structure.add_ref(bend_s(outter_s_bend_x,outter_s_bend_y,gf.cross_section.strip(width=coupler_width)))

        straight2_down = structure.add_ref(str_line(remaining_length, gf.cross_section.strip(width=coupler_width)))

        taper1_down.connect("o1", straight1_down.ports["o2"])
        s_bend1_down.connect("o1", taper1_down.ports["o2"])
        straight2_down.connect("o1", s_bend1_down.ports["o2"])
        coupler_structure.connect("o1", straight2_down.ports["o2"])

        #  **Upper Input Path**
        straight1_up = structure.add_ref(str_line(remaining_length, gf.cross_section.strip(width = coupler_width)))

        s_bend1_up = structure.add_ref(bend_s(outter_s_bend_x,outter_s_bend_y,gf.cross_section.strip(width=coupler_width)))

        s_bend1_up.mirror()  # Mirror for correct orientation

        taper1_up = structure.add_ref(taper(300,coupler_width,3.0,"strip"))

        straight2_up = structure.add_ref(str_line(1000,gf.cross_section.strip(width=3.0)))

        straight1_up.connect("o1", coupler_structure.ports["o2"])
        s_bend1_up.connect("o1", straight1_up.ports["o2"])
        taper1_up.connect("o1", s_bend1_up.ports["o2"])
        straight2_up.connect("o1", taper1_up.ports["o2"])

        #  **Lower Output Path**
        straight3_down = structure.add_ref(str_line(remaining_length,gf.cross_section.strip(width=coupler_width)))

        s_bend2_down = structure.add_ref(bend_s(outter_s_bend_x,outter_s_bend_y,gf.cross_section.strip(width=coupler_width)))

        s_bend2_down.mirror(p1=(0, 0), p2=(0, 1))

        taper2_down = structure.add_ref(taper(300,coupler_width,3.0,"strip"))

        straight4_down = structure.add_ref(str_line(1000,gf.cross_section.strip(width=3.0)))

        straight3_down.connect("o1", coupler_structure.ports["o4"])
        s_bend2_down.connect("o1", straight3_down.ports["o2"])
        taper2_down.connect("o1", s_bend2_down.ports["o2"])
        straight4_down.connect("o1", taper2_down.ports["o2"])

        #  **Upper Output Path**
        straight3_up = structure.add_ref(str_line(remaining_length,gf.cross_section.strip(width=coupler_width)))

        s_bend2_up = structure.add_ref(bend_s(outter_s_bend_x,outter_s_bend_y,gf.cross_section.strip(width=coupler_width)))

        taper2_up = structure.add_ref(taper(300,coupler_width,3.0,"strip"))

        straight4_up = structure.add_ref(str_line(1000,gf.cross_section.strip(width=3.0)))

        straight3_up.connect("o1", coupler_structure.ports["o3"])
        s_bend2_up.connect("o1", straight3_up.ports["o2"])
        taper2_up.connect("o1", s_bend2_up.ports["o2"])
        straight4_up.connect("o1", taper2_up.ports["o2"])

         # **Labels**
        structure.add_ref(text_on_structure("DC 50:50 with S-Bend",50,(straight1_down.xmin + 500, straight1_down.ymax + 150),"left","WG"))

        structure.add_ref(text_on_structure("DC 50:50 with S-Bend",50,(straight4_down.xmin - 500, straight4_down.ymax + 150),"left","WG"))

        #  **Assign Ports**
        structure.add_port(name=f"o{4*idx-3}", port=straight1_down.ports["o1"])
        structure.add_port(name=f"o{4*idx-2}", port=straight2_up.ports["o2"])
        structure.add_port(name=f"o{4*idx-1}", port=straight4_up.ports["o2"])
        structure.add_port(name=f"o{4*idx}", port=straight4_down.ports["o2"])

        ref = c.add_ref(structure)
        ref.move((initial_pos_x,initial_pos_y + idx * y_offset))

    x_final_pos = 0
    y_final_pos = c.ymax

    return c,x_final_pos,y_final_pos

def multiple_directional_Couplers_25_to_75_with_SBend(num_of_couplers,
                                                      initial_pos_x,
                                                      initial_pos_y,
                                                      y_offset,
                                                      coupler_width,
                                                      gap,
                                                      inner_s_bend_x,
                                                      inner_s_bend_y,
                                                      outter_s_bend_x,
                                                      outter_s_bend_y,  # S-Bend (width, height)
                                                      laser_lambda,
                                                      n1,
                                                      n2,
                                                      chip_length=11000,  # Total allowed chip length
                                                      margin=200          # Safe margins
                                                      ):
    """
    Generates directional couplers with correctly oriented S-Bends at input and output.

    Returns:
        gf.Component: Multiple directional couplers with properly adjusted S-Bends.
    """

    if gap < 0.4:
        raise ValueError(f"Invalid gap: {gap} µm. Must be >= 0.4 µm.")

    c = gf.Component()
    # Example base_length (approximation)
    base_length = laser_lambda / ( 3* (n1 - n2)) # L = lamba/(4 * (delta n)) para 780 nm , para un acoplamiento de 3/4

    for idx in range(1, num_of_couplers+1):
        structure = gf.Component()
        length = base_length

        total_fixed_length = (
                2 * (1000 + 300)  # waveguides + tapers
                + length
                + 2 * outter_s_bend_x
        )

        usable_length = chip_length - 2 * margin
        remaining_length = max((usable_length - total_fixed_length) / 2, 100)

        coupler_structure = structure.add_ref(coupler(gap,length,inner_s_bend_x,inner_s_bend_y,gf.cross_section.strip(width=coupler_width)))

        #  **Lower Input Path**
        straight1_down = structure.add_ref( str_line(1000, gf.cross_section.strip(width=3.0)) )

        taper1_down = structure.add_ref(taper(300,3.0,coupler_width,"strip"))

        s_bend1_down = structure.add_ref(bend_s(outter_s_bend_x,outter_s_bend_y,gf.cross_section.strip(width=coupler_width)))

        straight2_down = structure.add_ref(str_line(remaining_length, gf.cross_section.strip(width=coupler_width)))

        taper1_down.connect("o1", straight1_down.ports["o2"])
        s_bend1_down.connect("o1", taper1_down.ports["o2"])
        straight2_down.connect("o1", s_bend1_down.ports["o2"])
        coupler_structure.connect("o1", straight2_down.ports["o2"])

        #  **Upper Input Path**
        straight1_up = structure.add_ref(str_line(remaining_length, gf.cross_section.strip(width = coupler_width)))

        s_bend1_up = structure.add_ref(bend_s(outter_s_bend_x,outter_s_bend_y,gf.cross_section.strip(width=coupler_width)))

        s_bend1_up.mirror()  # Mirror for correct orientation

        taper1_up = structure.add_ref(taper(300,coupler_width,3.0,"strip"))

        straight2_up = structure.add_ref(str_line(1000,gf.cross_section.strip(width=3.0)))

        straight1_up.connect("o1", coupler_structure.ports["o2"])
        s_bend1_up.connect("o1", straight1_up.ports["o2"])
        taper1_up.connect("o1", s_bend1_up.ports["o2"])
        straight2_up.connect("o1", taper1_up.ports["o2"])

        #  **Lower Output Path**
        straight3_down = structure.add_ref(str_line(remaining_length,gf.cross_section.strip(width=coupler_width)))

        s_bend2_down = structure.add_ref(bend_s(outter_s_bend_x,outter_s_bend_y,gf.cross_section.strip(width=coupler_width)))

        s_bend2_down.mirror(p1=(0, 0), p2=(0, 1))

        taper2_down = structure.add_ref(taper(300,coupler_width,3.0,"strip"))

        straight4_down = structure.add_ref(str_line(1000,gf.cross_section.strip(width=3.0)))

        straight3_down.connect("o1", coupler_structure.ports["o4"])
        s_bend2_down.connect("o1", straight3_down.ports["o2"])
        taper2_down.connect("o1", s_bend2_down.ports["o2"])
        straight4_down.connect("o1", taper2_down.ports["o2"])

        #  **Upper Output Path**
        straight3_up = structure.add_ref(str_line(remaining_length,gf.cross_section.strip(width=coupler_width)))

        s_bend2_up = structure.add_ref(bend_s(outter_s_bend_x,outter_s_bend_y,gf.cross_section.strip(width=coupler_width)))

        taper2_up = structure.add_ref(taper(300,coupler_width,3.0,"strip"))

        straight4_up = structure.add_ref(str_line(1000,gf.cross_section.strip(width=3.0)))

        straight3_up.connect("o1", coupler_structure.ports["o3"])
        s_bend2_up.connect("o1", straight3_up.ports["o2"])
        taper2_up.connect("o1", s_bend2_up.ports["o2"])
        straight4_up.connect("o1", taper2_up.ports["o2"])

         # **Labels**
        structure.add_ref(text_on_structure("DC 25:75 with S-Bend",50,(straight1_down.xmin + 500, straight1_down.ymax + 150),"left","WG"))

        structure.add_ref(text_on_structure("DC 25:75 with S-Bend",50,(straight4_down.xmin - 500, straight4_down.ymax + 150),"left","WG"))

        #  **Assign Ports**
        structure.add_port(name=f"o{4*idx-3}", port=straight1_down.ports["o1"])
        structure.add_port(name=f"o{4*idx-2}", port=straight2_up.ports["o2"])
        structure.add_port(name=f"o{4*idx-1}", port=straight4_up.ports["o2"])
        structure.add_port(name=f"o{4*idx}", port=straight4_down.ports["o2"])

        ref = c.add_ref(structure)
        ref.move((initial_pos_x,initial_pos_y + idx * y_offset))

    x_final_pos = 0
    y_final_pos = c.ymax

    return c, x_final_pos, y_final_pos

def multiple_directional_Couplers_0_to_100_with_SBend(num_of_couplers,
                                                      initial_pos_x,
                                                      initial_pos_y,
                                                      y_offset,
                                                      coupler_width,
                                                      gap,
                                                      inner_s_bend_x,
                                                      inner_s_bend_y,
                                                      outter_s_bend_x,
                                                      outter_s_bend_y,  # S-Bend (width, height)
                                                      laser_lambda,
                                                      n1,
                                                      n2,
                                                      chip_length=11000,  # Total allowed chip length
                                                      margin=200          # Safe margins
                                                      ):
    """
    Generates directional couplers with correctly oriented S-Bends at input and output.

    Returns:
        gf.Component: Multiple directional couplers with properly adjusted S-Bends.
    """

    if gap < 0.4:
        raise ValueError(f"Invalid gap: {gap} µm. Must be >= 0.4 µm.")

    c = gf.Component()
    # Example base_length (approximation)
    base_length = laser_lambda / ( 2 * (n1 - n2)) # L = lamba/(4 * (delta n)) para 780 nm , para un acoplamiento de 1

    for idx in range(1, num_of_couplers+1):
        structure = gf.Component()
        length = base_length

        total_fixed_length = (
                2 * (1000 + 300)  # waveguides + tapers
                + length
                + 2 * outter_s_bend_x
        )

        usable_length = chip_length - 2 * margin
        remaining_length = max((usable_length - total_fixed_length) / 2, 100)

        coupler_structure = structure.add_ref(coupler(gap,length,inner_s_bend_x,inner_s_bend_y,gf.cross_section.strip(width=coupler_width)))

        #  **Lower Input Path**
        straight1_down = structure.add_ref( str_line(1000, gf.cross_section.strip(width=3.0)) )

        taper1_down = structure.add_ref(taper(300,3.0,coupler_width,"strip"))

        s_bend1_down = structure.add_ref(bend_s(outter_s_bend_x,outter_s_bend_y,gf.cross_section.strip(width=coupler_width)))

        straight2_down = structure.add_ref(str_line(remaining_length, gf.cross_section.strip(width=coupler_width)))

        taper1_down.connect("o1", straight1_down.ports["o2"])
        s_bend1_down.connect("o1", taper1_down.ports["o2"])
        straight2_down.connect("o1", s_bend1_down.ports["o2"])
        coupler_structure.connect("o1", straight2_down.ports["o2"])

        #  **Upper Input Path**
        straight1_up = structure.add_ref(str_line(remaining_length, gf.cross_section.strip(width = coupler_width)))

        s_bend1_up = structure.add_ref(bend_s(outter_s_bend_x,outter_s_bend_y,gf.cross_section.strip(width=coupler_width)))

        s_bend1_up.mirror()  # Mirror for correct orientation

        taper1_up = structure.add_ref(taper(300,coupler_width,3.0,"strip"))

        straight2_up = structure.add_ref(str_line(1000,gf.cross_section.strip(width=3.0)))

        straight1_up.connect("o1", coupler_structure.ports["o2"])
        s_bend1_up.connect("o1", straight1_up.ports["o2"])
        taper1_up.connect("o1", s_bend1_up.ports["o2"])
        straight2_up.connect("o1", taper1_up.ports["o2"])

        #  **Lower Output Path**
        straight3_down = structure.add_ref(str_line(remaining_length,gf.cross_section.strip(width=coupler_width)))

        s_bend2_down = structure.add_ref(bend_s(outter_s_bend_x,outter_s_bend_y,gf.cross_section.strip(width=coupler_width)))

        s_bend2_down.mirror(p1=(0, 0), p2=(0, 1))

        taper2_down = structure.add_ref(taper(300,coupler_width,3.0,"strip"))

        straight4_down = structure.add_ref(str_line(1000,gf.cross_section.strip(width=3.0)))

        straight3_down.connect("o1", coupler_structure.ports["o4"])
        s_bend2_down.connect("o1", straight3_down.ports["o2"])
        taper2_down.connect("o1", s_bend2_down.ports["o2"])
        straight4_down.connect("o1", taper2_down.ports["o2"])

        #  **Upper Output Path**
        straight3_up = structure.add_ref(str_line(remaining_length,gf.cross_section.strip(width=coupler_width)))

        s_bend2_up = structure.add_ref(bend_s(outter_s_bend_x,outter_s_bend_y,gf.cross_section.strip(width=coupler_width)))

        taper2_up = structure.add_ref(taper(300,coupler_width,3.0,"strip"))

        straight4_up = structure.add_ref(str_line(1000,gf.cross_section.strip(width=3.0)))

        straight3_up.connect("o1", coupler_structure.ports["o3"])
        s_bend2_up.connect("o1", straight3_up.ports["o2"])
        taper2_up.connect("o1", s_bend2_up.ports["o2"])
        straight4_up.connect("o1", taper2_up.ports["o2"])

         # **Labels**
        structure.add_ref(text_on_structure("DC 0:100 with S-Bend",50,(straight1_down.xmin + 500, straight1_down.ymax + 150),"left","WG"))

        structure.add_ref(text_on_structure("DC 0:100 with S-Bend",50,(straight4_down.xmin - 500, straight4_down.ymax + 150),"left","WG"))

        #  **Assign Ports**
        structure.add_port(name=f"o{4*idx-3}", port=straight1_down.ports["o1"])
        structure.add_port(name=f"o{4*idx-2}", port=straight2_up.ports["o2"])
        structure.add_port(name=f"o{4*idx-1}", port=straight4_up.ports["o2"])
        structure.add_port(name=f"o{4*idx}", port=straight4_down.ports["o2"])

        ref = c.add_ref(structure)
        ref.move((initial_pos_x,initial_pos_y + idx * y_offset))

    x_final_pos = 0
    y_final_pos = c.ymax

    return c, x_final_pos, y_final_pos