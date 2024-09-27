document.addEventListener("DOMContentLoaded", function () {
  // Chart settings
  const chartSettings = {
    width: parseInt(d3.select(".dash-item").style("width")),
    height: 300,
    margin: { top: 20, right: 40, bottom: 30, left: 40 },
  };

  function createLineChart(data, selector, yLabel, color, unit) {
    const svg = d3
      .select(selector)
      .append("svg")
      .attr("viewBox", `0 0 ${chartSettings.width} ${chartSettings.height}`)
      .attr("preserveAspectRatio", "xMidYMid meet")
      .attr("style", "max-width: 100%; height: auto;");

    const gradientId = `gradient-${selector.replace("#", "")}`;
    const defs = svg.append("defs");
    const gradient = defs
      .append("linearGradient")
      .attr("id", gradientId)
      .attr("x1", "0%")
      .attr("y1", "0%")
      .attr("x2", "0%")
      .attr("y2", "100%");

    gradient
      .append("stop")
      .attr("offset", "0%")
      .attr("stop-color", color)
      .attr("stop-opacity", 0.2);

    gradient
      .append("stop")
      .attr("offset", "100%")
      .attr("stop-color", color)
      .attr("stop-opacity", 0);

    const x = d3
      .scaleTime()
      .domain(d3.extent(data, (d) => d.date))
      .range([
        chartSettings.margin.left,
        chartSettings.width - chartSettings.margin.right,
      ]);

    const y = d3
      .scaleLinear()
      .domain([
        d3.min(data, (d) => d.value) * 0.95,
        d3.max(data, (d) => d.value),
      ])
      .nice()
      .range([
        chartSettings.height - chartSettings.margin.bottom,
        chartSettings.margin.top,
      ]);

    const line = d3
      .line()
      .x((d) => x(d.date))
      .y((d) => y(d.value));

    const area = d3
      .area()
      .x((d) => x(d.date))
      .y0(chartSettings.height - chartSettings.margin.bottom)
      .y1((d) => y(d.value));

    svg
      .append("g")
      .attr(
        "transform",
        `translate(0,${chartSettings.height - chartSettings.margin.bottom})`
      )
      .call(
        d3
          .axisBottom(x)
          .ticks(chartSettings.width / 80)
          .tickSizeOuter(0)
      );

    const yAxis = svg
      .append("g")
      .attr("transform", `translate(${chartSettings.margin.left},0)`)
      .call(d3.axisLeft(y).ticks(5))
      .call((g) =>
        g
          .append("text")
          .attr("x", -chartSettings.margin.left)
          .attr("y", 0)
          .attr("fill", "currentColor")
          .attr("text-anchor", "start")
          .attr("font-size", "14px")
          .text(yLabel)
      );

    // Add gridlines
    yAxis
      .selectAll(".tick line")
      .clone()
      .attr(
        "x2",
        chartSettings.width -
          chartSettings.margin.left -
          chartSettings.margin.right
      )
      .attr("stroke", "#ddd")
      .attr("stroke-width", 0.7)
      .attr("stroke-dasharray", "2,2");

    svg
      .append("line")
      .attr("x1", chartSettings.margin.left)
      .attr("y1", chartSettings.height - chartSettings.margin.bottom)
      .attr("x2", chartSettings.width - chartSettings.margin.right)
      .attr("y2", chartSettings.height - chartSettings.margin.bottom)
      .attr("stroke", "black");

    svg
      .append("line")
      .attr("x1", chartSettings.margin.left)
      .attr("y1", chartSettings.margin.top)
      .attr("x2", chartSettings.margin.left)
      .attr("y2", chartSettings.height - chartSettings.margin.bottom)
      .attr("stroke", "black");

    // Append the area before the line
    svg
      .append("path")
      .datum(data)
      .attr("fill", `url(#${gradientId})`)
      .attr("d", area);

    const path = svg
      .append("path")
      .datum(data)
      .attr("fill", "none")
      .attr("stroke", color)
      .attr("stroke-width", 2.5)
      .attr("d", line);

    const totalLength = path.node().getTotalLength();
    path
      .attr("stroke-dasharray", `${totalLength} ${totalLength}`)
      .attr("stroke-dashoffset", totalLength)
      .transition()
      .duration(5000)
      .ease(d3.easeLinear)
      .attr("stroke-dashoffset", 0);

    // Display the most recent data point
    const latestData = data[data.length - 1];
    svg
      .append("text")
      .attr("x", chartSettings.width - chartSettings.margin.right)
      .attr("y", chartSettings.margin.top)
      .attr("text-anchor", "end")
      .attr("font-size", "12px")
      .attr("font-family", "Arial, sans-serif")
      .attr("fill", "black")
      .text(
        `Latest: ${d3.timeFormat("%b %d, %Y")(
          latestData.date
        )} - ${latestData.value.toFixed(2)}${unit}`
      );

    const tooltip = svg.append("g").style("display", "none");

    const tooltipBackground = tooltip
      .append("rect")
      .attr("x", -65)
      .attr("y", -5)
      .attr("width", 90)
      .attr("height", 32)
      .attr("rx", 0)
      .attr("fill", "white")
      .attr("stroke", "black")
      .attr("stroke-width", 0.5)
      .style("opacity", 0.8);

    const tooltipCircle = tooltip
      .append("circle")
      .attr("r", 3)
      .attr("fill", "black")
      .attr("stroke", "black")
      .attr("stroke-width", 1);

    const tooltipText = tooltip
      .append("text")
      .attr("x", -60)
      .attr("y", 7.5)
      .attr("font-size", 12)
      .attr("font-family", "Arial, sans-serif")
      .attr("text-anchor", "start")
      .attr("fill", "black");

    const bisect = d3.bisector((d) => d.date).left;

    const tooltipLine = svg
      .append("line")
      .attr("stroke", "black")
      .attr("stroke-width", 1)
      .style("stroke-dasharray", "3,3")
      .style("opacity", 0);

    function pointermoved(event) {
      const [xm] = d3.pointer(event);
      const xDate = x.invert(xm);
      const index = bisect(data, xDate);
      const d0 = data[index - 1];
      const d1 = data[index];
      const d = xDate - d0.date > d1.date - xDate ? d1 : d0;

      tooltip.style("display", null);
      tooltipLine.style("opacity", 1);

      const xPosition = x(d.date);
      const yPosition = y(d.value);

      tooltipCircle.attr("cx", xPosition).attr("cy", yPosition);
      tooltipBackground.attr("x", xPosition - 45);
      tooltipText.selectAll("tspan").remove();
      tooltipText
        .attr("x", xPosition - 40)
        .append("tspan")
        .text(`${d.value.toFixed(2)}${unit}`)
        .attr("x", xPosition - 40)
        .attr("dy", 0);
      tooltipText
        .append("tspan")
        .text(`${d3.timeFormat("%b %d, %Y")(d.date)}`)
        .attr("x", xPosition - 40)
        .attr("dy", 15);

      tooltipLine
        .attr("x1", xPosition)
        .attr("x2", xPosition)
        .attr("y1", chartSettings.margin.top + 10)
        .attr("y2", chartSettings.height - chartSettings.margin.bottom);
    }

    function pointerleft() {
      tooltip.style("display", "none");
      tooltipLine.style("opacity", 0);
    }

    svg
      .append("rect")
      .attr("fill", "none")
      .attr("pointer-events", "all")
      .attr("width", chartSettings.width)
      .attr("height", chartSettings.height)
      .on("mousemove", pointermoved)
      .on("mouseleave", pointerleft);

    svg.on("mouseenter", pointermoved).on("mouseleave", pointerleft);
  }

  function parseDateUTC(dateString) {
    // Parse the date string and create a UTC date
    const date = new Date(dateString);
    return new Date(
      Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate())
    );
  }

  d3.json("https://seal-app-scm6d.ondigitalocean.app/api/gdp")
    .then((data) => {
      const parsedData = data.map((d) => ({
        date: new parseDateUTC(d.date),
        value: +d.value,
      }));
      createLineChart(parsedData, "#gdp-chart", "GDP ($)", "navy", "$");
    })
    .catch((error) => console.error("Error loading data:", error));

  d3.json("https://seal-app-scm6d.ondigitalocean.app/api/unemployment")
    .then((data) => {
      const parsedData = data.map((d) => ({
        date: new parseDateUTC(d.date),
        value: +d.value,
      }));
      createLineChart(
        parsedData,
        "#unemployment-chart",
        "Unemployment (%)",
        "Crimson",
        "%"
      );
    })
    .catch((error) => console.error("Error loading data:", error));

  d3.json("https://seal-app-scm6d.ondigitalocean.app/api/inflation")
    .then((data) => {
      const parsedData = data.map((d) => ({
        date: new parseDateUTC(d.date),
        value: +d.value,
      }));
      createLineChart(
        parsedData,
        "#inflation-chart",
        "Inflation (%)",
        "SeaGreen",
        "%"
      );
    })
    .catch((error) => console.error("Error loading data:", error));

  d3.json("https://seal-app-scm6d.ondigitalocean.app/api/interest-rates")
    .then((data) => {
      const parsedData = data.map((d) => ({
        date: new parseDateUTC(d.date),
        value: +d.value,
      }));
      createLineChart(
        parsedData,
        "#interest-rates-chart",
        "Interest Rates (%)",
        "#6495ED",
        "%"
      );
    })
    .catch((error) => console.error("Error loading data:", error));

  d3.json("https://seal-app-scm6d.ondigitalocean.app/api/sp500")
    .then((data) => {
      const parsedData = data.map((d) => ({
        date: new parseDateUTC(d.date),
        value: +d.close,
      }));
      createLineChart(
        parsedData,
        "#sp500-chart",
        "S&P 500 Index",
        "orange",
        "$"
      );
    })
    .catch((error) => console.error("Error loading data:", error));

  d3.json("https://seal-app-scm6d.ondigitalocean.app/api/exchange-rates")
    .then((data) => {
      const parsedData = data.map((d) => ({
        date: new parseDateUTC(d.date),
        value: +d.close,
      }));
      createLineChart(
        parsedData,
        "#exchange-rates-chart",
        "USD/EUR",
        "LightSeaGreen",
        "$"
      );
    })
    .catch((error) => console.error("Error loading data:", error));
});
