"""
Module to generate the Latex source files for bubble plots.

A plot has three facets, one as the y axis and two on the x axis.
"""

import colorsys
import csv
import os
from itertools import zip_longest
from string import Template
from typing import Dict, NamedTuple, Sequence, Set, Tuple


class Facets(NamedTuple):
    """
    Describe the name of the facets of a bubble plot.

    Attributes
    ----------
    y: str
        Name of the facet on the y axis.
    x_left: str
        Name of the facet on the left part of the x axis.
    x_right: str
        Name of the facet on the right part of the x axis.
    """

    y: str
    x_left: str
    x_right: str


class Config(NamedTuple):
    """
    Describe custom settings for the bubble plot.

    Attributes
    ----------
    x_left_offset: int
        Define the offset at which the first x tick near the left of the y axis is placed.
    x_right_offset: int
        Define the offset at which the first x tick near the right of the y axis is placed.
    class_year: str
        Label representing the publication year.
    field_names: Sequence[str]
        Name of the columns for the output file of the bubble plot.
    latex_template: str
        Path to the Latex template.
    output_dir: str
        Path where to save the generated files (i.e., CSV & TeX).
    color_map: Sequence[Tuple[float, float, float]]
        Colour map for years.
    """

    x_left_offset: int
    x_right_offset: int
    class_year: str
    field_names: Sequence[str]
    latex_template: str
    output_dir: str
    color_map: Sequence[Tuple[float, float, float]]


class Occurrence:
    """
    Compute the occurrence of a bubble.

    Attributes
    ----------
    occurrence: int
        Occurrence of a bubble.
    year: str
        Earliest year from entries related to the bubble.
    """

    __slots__ = ("occurrence", "year")

    def __init__(self, year: str):
        """
        Parameters
        ----------
        year: str
            Year of the first entry related to the bubble.
        """
        self.occurrence: int = 1
        self.year: str = year

    def update(self, new_year: str):
        """
        Update the occurrences and save the earliest publish year.

        Parameters
        ----------
        new_year: str
            New publish year to update `year` with if it's less than the current year.
        """
        self.occurrence += 1
        self.year = min(self.year, new_year)

    def __str__(self):
        return f"Occurrence: {self.occurrence}, Year: {self.year}"

    def __repr__(self):
        return f"{self.__class__.__name__}{(self.occurrence, self.year)}"


class Bubble(NamedTuple):
    """
    Bubble label on the x axis and y axis.

    Attributes
    ----------
    label_x: str
        Label displayed as a x tick label.
    label_y: str
        Label displayed as a y tick label.

    Notes
    -----
    The x and y tick label are keys from the Latex
    `pgfplots <https://ctan.org/pkg/pgfplots>`_ package.
    """

    label_x: str
    label_y: str


class SplitXAxis(NamedTuple):
    """
    Facet on one side of the splitted x axis.

    Attributes
    ----------
    facet: str
        Name of the facet.
    bubbles: Dict[Bubble, Occurence]
        Bubbles related to the facet
    """

    facet: str
    bubbles: Dict[Bubble, Occurrence]

    def update(self, entry: Dict[str, str], year: str, y_axis: str) -> None:
        """
        Update the bubble occurrences related to the side of the x axis.

        Parameters
        ----------
        entry: Dict[str, str]
            Entries to take as input for the occurrence computation.
        year: str
            Publication year of the entry.
        y_axis: str
            Name of the facet of the y axis.
        """
        try:
            label_x = entry[self.facet]
        except KeyError as error:
            raise KeyError(
                f"Unknown facet named: {error} on the x axis"
            ) from error
        try:
            label_y = entry[y_axis]
        except KeyError as error:
            raise KeyError(
                f"Unknown facet named: {error} on the y axis"
            ) from error

        bubble = Bubble(label_x=label_x, label_y=label_y)
        if bubble in self.bubbles:
            self.bubbles[bubble].update(year)
        else:
            self.bubbles[bubble] = Occurrence(year)


class XAxis(NamedTuple):
    """
    Facets of the x axis.

    Attributes
    ----------
    left: SplitXAxis
        Left facet of a bubble plot.
    right: SplitXAxis
        Right facet of a bubble plot.
    """

    left: SplitXAxis
    right: SplitXAxis


class BubblePlot:
    """
    Describe a bubble plot with 3 facets.

    Attributes
    ----------
    x_axis: XAxis
        Facets on the x axis. One to the left of the y axis and the other to the right.
    y_axis: str
        Facet on the y axis.
    """

    __slots__ = ("x_axis", "y_axis")

    def __init__(self, facets: Facets):
        """
        Parameters
        ----------
        facets: Facets
            Describe the facets label of the bubble plot.
        """
        y_axis, x_left_axis, x_right_axis = facets
        self.x_axis: XAxis = XAxis(
            left=SplitXAxis(facet=x_left_axis, bubbles={}),
            right=SplitXAxis(facet=x_right_axis, bubbles={}),
        )
        self.y_axis: str = y_axis

    def __str__(self):
        """
        Returns
        -------
        str
            The name of the left facet on the x axis,
            the name of the facet on the y axis, and
            the name of the right facet on the x axis.
        """
        return (
            f"{self.x_axis.left.facet}_{self.y_axis}_{self.x_axis.right.facet}"
        )

    def __repr__(self):
        return f"{self.__class__.__name__}{(self.x_axis, self.y_axis)}"

    def update(self, entry: Dict[str, str], year: str) -> None:
        """
        Update the bubbles occurrences of the plot.

        Parameters
        ----------
        entry: Dict[str, str]
            Label and value to use as input for a bubble.
        year: str
            Publication year of the entry.
        conf: Config
            Describe custom settings for the plots.
        """
        self.x_axis.left.update(entry, year, self.y_axis)
        self.x_axis.right.update(entry, year, self.y_axis)


class CSVWriter(NamedTuple):
    """
    Save a bubble plot as a CSV file.

    Attributes
    ----------
    plot: BubblePlot
        Plot to prepare and format.
    years: Set[str]
        Publication years.
    conf: Config
        Describe custom settings for the plots.
    """

    plot: BubblePlot
    years: Set[str]
    conf: Config

    def compute_year_score_mapping(self) -> Dict[str, int]:
        """
        Compute the mapping between years and colour map scores. The scores are mapped in
        chronological order of the years.

        Returns
        -------
        Dict[str, str]
            Mapping between a year and a score.

        Notes
        -----
        The colour map score is in the interval [0-1000].
        See paragraph ``Colormap Input Format Reference`` in Section 4.7.6 of the
        Latex `pgfplots <https://ctan.org/pkg/pgfplots>`_ package.
        """
        idx = 0
        incr = 1000 // (len(self.years) - 1)
        year_score = {}
        for year in sorted(self.years):
            year_score[year] = idx
            idx += incr
        return year_score

    def compute_labels_indices_mapping(
        self,
    ) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
        """
        Compute the mapping between labels and their indices.

        Returns
        -------
        Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]
            The mapping between:
                - the left x tick labels and their indices.
                - the right x tick labels and their indices.
                - the y tick labels and their indices.
        """

        labels_x_left = sorted(
            set(b.label_x for b in self.plot.x_axis.left.bubbles)
        )
        labels_x_left_len = len(labels_x_left)
        # From -N to -conf.x_left_offset
        x_left_mapping = {
            label: -(labels_x_left_len - i + self.conf.x_left_offset)
            for i, label in enumerate(labels_x_left)
        }

        labels_x_right = sorted(
            set(b.label_x for b in self.plot.x_axis.right.bubbles)
        )
        # From conf.x_left_offset to N
        x_right_mapping = {
            label: i + self.conf.x_right_offset
            for i, label in enumerate(labels_x_right)
        }

        labels_y = sorted(
            set(
                b.label_y
                for b in list(self.plot.x_axis.right.bubbles)
                + list(self.plot.x_axis.left.bubbles)
            )
        )
        # From 0 to N-1
        y_mapping = {label: i for i, label in enumerate(labels_y)}

        return x_left_mapping, x_right_mapping, y_mapping

    def prepared_bubbles_data(
        self,
        x_left_mapping: Dict[str, int],
        x_right_mapping: Dict[str, int],
        y_mapping: Dict[str, int],
        year_mapping: Dict[str, int],
    ) -> Sequence[Tuple[int, int, int, int]]:
        """
        Prepare bubbles data.

        Parameters
        ----------
        x_left_mapping: Dict[str, int]
            The left x tick labels and their indices.
        x_right_mapping: Dict[str, int]
            The right x tick labels and their indices.
        y_mapping: Dict[str, int]
            The y tick labels and their indices.
        year_mapping: Dict[str, int]
            The publication years and their score.

        Returns
        -------
        Sequence[Tuple[int, int, int, int]]
            Lines of the CSV file representing each bubble.
            The columns are, in order:
            - The index of the y tick label related to the bubble.
            - The index of the x tick label related to the bubble.
            - The number of occurrence of the bubble.
            - The earliest publication year related to the bubble.
        """
        axis_mapping = (
            (self.plot.x_axis.left, x_left_mapping),
            (self.plot.x_axis.right, x_right_mapping),
        )
        return [
            (
                y_mapping[bubble.label_y],
                x_mapping[bubble.label_x],
                occurence.occurrence,
                year_mapping[occurence.year],
            )
            for axis, x_mapping in axis_mapping
            for bubble, occurence in axis.bubbles.items()
        ]

    def write(
        self,
        data: Sequence[Tuple[int, int, int, int]],
        labels_y: Sequence[str],
        labels_x: Sequence[str],
    ) -> None:
        """
        Save bubble plot as a CVS file.

        Parameters
        ----------
        data: Sequence[Tuple[int, int, int, int]]
             Bubble plot data prepared.
        labels_y: Sequence[str]
            Labels of the facet on the y axis.
        labels_x: Sequence[str]
            Labels of the facets on the x axis.
        """

        # TODO: Ugly
        y_indices = []
        x_indices = []
        occurences = []
        year_scores = []
        for y_idx, x_idx, occ, y_s in data:
            y_indices.append(y_idx)
            x_indices.append(x_idx)
            occurences.append(occ)
            year_scores.append(y_s)

        with open(
            os.path.join(self.conf.output_dir, f"{self.plot}.csv"), "w"
        ) as output_file:
            writer = csv.writer(output_file)
            writer.writerow(self.conf.field_names)
            writer.writerows(
                zip_longest(
                    y_indices,
                    x_indices,
                    occurences,
                    year_scores,
                    labels_y,
                    labels_x,
                )
            )

    def save_plot(self) -> None:
        """
        Compute the mapping between labels and their indices; year and their scores.

        Prepared bubble plot data for the CSV file. And write data and
        labels into a file.
        """
        (
            x_left_mapping,
            x_right_mapping,
            y_mapping,
        ) = self.compute_labels_indices_mapping()
        year_score = self.compute_year_score_mapping()
        bubbles_data = self.prepared_bubbles_data(
            x_left_mapping, x_right_mapping, y_mapping, year_score
        )
        sorted_bubbles_data = sorted(bubbles_data, key=lambda t: t[1])
        self.write(
            sorted_bubbles_data,
            list(y_mapping.keys()),
            list(x_left_mapping.keys()) + list(x_right_mapping.keys()),
        )


def compute_color_map(years_len: int) -> Sequence[Tuple[float, float, float]]:
    """
    Affect a distinct (as much as possible) colour to each year.

    Parameters
    ----------
    years_len: int
        Number of publication years.

    Returns
    -------
    Sequence[Tuple[float, float, float]]
        Colours used for bubbles in the plot.
    """
    hsv_tuples = [(i * 1.0 / years_len, 0.5, 1) for i in range(years_len)]
    return tuple(colorsys.hsv_to_rgb(*hsv) for hsv in hsv_tuples)


class LatexBubblePlotWriter:
    """
    Prepare the Latex template for a bubble plot.

    Attributes
    ----------
    plot: BubblePlot
        Plot to prepare and format.
    years: Sequence[str]
        Publication years.
    conf: Config
        Describe custom settings for the plots.
    x_indices: Sequence[int]
        Indices of bubbles on the x axis.
    """

    __slots__ = ("plot", "years", "year_color", "conf", "x_indices")

    def __init__(self, plot: BubblePlot, years: Set[str], conf: Config):
        """
        Parameters
        ----------
        plot: BubblePlot
            Plot to prepare and format.
        years: Set[str]
            Publication years.
        conf: Config
            Describe custom settings for the plots.
        """
        self.plot: BubblePlot = plot
        self.years: Sequence[str] = sorted(years)
        self.conf: Config = conf
        with open(
            os.path.join(self.conf.output_dir, f"{plot}.csv"), "r"
        ) as plot_data:
            reader = csv.DictReader(plot_data)
            x_indice_field = conf.field_names[1]
            self.x_indices: Sequence[int] = tuple(
                int(row[x_indice_field]) for row in reader
            )

    def prepare_values(
        self, color_map: Dict[str, Tuple[float, float, float]]
    ) -> Dict[str, str]:
        """
        Prepare the values filled in the Latex template.

        Parameters
        ----------
        color_map: Dict[str, Tuple[float, float, float]]
            Mapping between a year and the related colour used for a bubble on the plot.

        Returns
        -------
        Dict[str, str]
            Values to fill the Latex template with.
        """
        delete_parenthesis = {ord(c): None for c in "[]"}
        return {
            "defineColorsYear": "\n".join(
                (
                    f"\\definecolor{{{year}}}{{rgb}}"
                    f"{{{str(color_map[year]).translate(delete_parenthesis)}}}"
                )
                for year in self.years
            ),
            "setColorsYear": "\n    ".join(
                f"color=({year})," for year in self.years
            ),
            "xMin": str(min(map(int, self.x_indices))),
            "xMax": str(max(map(int, self.x_indices))),
            "yLabel": self.plot.y_axis,
            "meta": self.conf.field_names[2],
            "xField": self.conf.field_names[5],
            "xIndexField": self.conf.field_names[1],
            "yField": self.conf.field_names[4],
            "yIndexField": self.conf.field_names[0],
            "yearField": self.conf.field_names[3],
            "xLeftLabel": self.plot.x_axis.left.facet,
            "xRightLabel": self.plot.x_axis.right.facet,
            "CSVDataFile": f"{self.plot}.csv",
            "colorsYear": ", ".join(self.years),
        }

    def write(self, template_values: Dict[str, str]) -> None:
        """
        Fill the Latex template with the bubble plot specific values.

        Parameters
        ----------
        template_values: Dict[str, str]
            Values to fill the Latex template with.
        """
        with open(self.conf.latex_template, "r") as latex_template:
            template = Template(latex_template.read())
            content = template.substitute(template_values)

        with open(
            os.path.join(self.conf.output_dir, f"{self.plot}.tex"), "w"
        ) as latex_output:
            latex_output.write(content)

    def save_plot(self) -> None:
        """Prepare the bubble plot specific values and fill the Latex template with them."""
        years_len = len(self.years)
        if years_len <= len(self.conf.color_map):
            rgb_colours = self.conf.color_map[:years_len]
        else:
            rgb_colours = compute_color_map(years_len)
        year_color = dict(zip(self.years, rgb_colours))
        template_values = self.prepare_values(year_color)
        self.write(template_values)


def compute_occurences_from(
    entries: Sequence[Dict[str, str]], plot_plan: Facets, conf: Config
) -> Tuple[BubblePlot, Set[str]]:
    """
    Compute the occurrences of each value for the given labels.

    Parameters
    ----------
    entries: Sequence[Dict[str, str]]
        Labels and values to use as input for the bubble plot.
    plot_plan: Facets
        Describe the labels used for the facets of the plot.
    conf: Config
        Describe custom settings for the plots.

    Returns
    -------
    Tuple[BubblePlot, Set[str]]
        - The initialised bubble plot described by `plot_plan`.
        - The years related to each entry.
    """
    years = set()
    plot = BubblePlot(plot_plan)
    for entry in entries:
        year = entry[conf.class_year]
        plot.update(entry, year)
        years.add(year)
    return plot, years


def build_and_save_plots(
    entries: Sequence[Dict[str, str]],
    plot_plans: Sequence[Facets],
    conf: Config,
) -> None:
    """
    Build and save bubble plots as CSV files.

    Parameters
    ----------
    entries: Sequence[Dict[str, str]]
        Labels and values to use as input for the bubble plot.
    plot_plan: Facets
        Describe the labels used for the facets of the plot.
    conf: Config
        Describe custom settings for the plots.
    """
    for plot_plan in plot_plans:
        plot, years = compute_occurences_from(entries, plot_plan, conf)
        writer_csv = CSVWriter(plot, years, conf)
        writer_csv.save_plot()
        writer_latex = LatexBubblePlotWriter(plot, years, conf)
        writer_latex.save_plot()
