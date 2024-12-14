from PIL import Image, ImageTk
import numpy as np


class DiagramLayout:
    def __init__(self, canvas):
        self.canvas = canvas
        self.elements = {}
        self.x_positions = {
            "input": 100,
            "conv": 400,
            "relu_pool": 800,
            "flatten": 1050,
            "fc": 1300,
            "softmax": 1500,
        }
        self.y_center = 300
        self.top_y = 50

    def draw_base_diagram(self):
        # Input Image
        input_array = np.random.rand(28, 28) * 255
        input_array = input_array.astype(np.uint8)
        pil_img = Image.fromarray(input_array, mode="L")
        pil_img = pil_img.resize((280, 280), Image.NEAREST)
        self.input_photo = ImageTk.PhotoImage(pil_img)

        # Draw input image and text
        x_input = self.x_positions["input"]
        image_id = self.canvas.create_image(
            x_input, self.y_center, image=self.input_photo
        )
        self.elements[("input", "image")] = image_id

        val_y = self.y_center + 160
        val_text_id = self.canvas.create_text(
            x_input, val_y, text="Val=...", font=("Helvetica", 12), fill="black"
        )
        self.elements[("input", "text")] = val_text_id

        self.canvas.create_text(
            x_input,
            self.top_y,
            text="Input\n1x28x28",
            font=("Helvetica", 14, "bold"),
            justify="center",
            fill="#333",
        )

        # Conv layer visualization (8 filters) in a 2x4 grid
        x_conv = self.x_positions["conv"]
        self.canvas.create_text(
            x_conv,
            self.top_y,
            text="Conv\n(8 filters)",
            font=("Helvetica", 14, "bold"),
            justify="center",
            fill="#333",
        )

        filter_count = 8
        filter_img_size = 70
        cols = 4
        rows = 2
        spacing_x = 10
        spacing_y = 10
        total_width = cols * (filter_img_size + spacing_x) - spacing_x
        total_height = rows * (filter_img_size + spacing_y) - spacing_y
        start_x = x_conv - total_width / 15 + filter_img_size / 15
        start_y = self.y_center - total_height / 2 + filter_img_size / 2

        blank_array = np.zeros((filter_img_size, filter_img_size), dtype=np.uint8)
        blank_pil = Image.fromarray(blank_array, mode="L")
        blank_img = ImageTk.PhotoImage(blank_pil)

        for i in range(filter_count):
            r = i // cols
            c = i % cols
            fx = start_x + c * (filter_img_size + spacing_x)
            fy = start_y + r * (filter_img_size + spacing_y)
            img_id = self.canvas.create_image(fx, fy, image=blank_img)
            self.elements[("conv", f"filter_{i}")] = img_id

        # ReLU/Pool: place the before/after images horizontally at ReLU/Pool x, aligned with y_center
        x_relu_pool = self.x_positions["relu_pool"]
        self.canvas.create_text(
            x_relu_pool,
            self.top_y,
            text="ReLU/Pool",
            font=("Helvetica", 14, "bold"),
            justify="center",
            fill="#333",
        )

        # ReLU placeholders (before/after) side by side
        relu_img_size = 50
        blank_relu = Image.new("L", (relu_img_size, relu_img_size), 127)
        relu_before = ImageTk.PhotoImage(blank_relu)
        relu_after = ImageTk.PhotoImage(blank_relu)
        relu_x_start = x_relu_pool - relu_img_size - 10
        relu_y = self.y_center - 30
        before_id = self.canvas.create_image(relu_x_start, relu_y, image=relu_before)
        after_id = self.canvas.create_image(
            relu_x_start + relu_img_size + 10, relu_y, image=relu_after
        )
        self.elements[("relu_pool", "before")] = before_id
        self.elements[("relu_pool", "after")] = after_id

        # Pooling: arrow from bigger box to smaller box below the ReLU images
        pool_y = self.y_center + 40
        big_box = self.canvas.create_rectangle(
            x_relu_pool - 30, pool_y - 25, x_relu_pool + 30, pool_y + 25, fill="#808080"
        )
        arrow_id = self.canvas.create_line(
            x_relu_pool + 40, pool_y, x_relu_pool + 80, pool_y, arrow="last", width=2
        )
        small_box = self.canvas.create_rectangle(
            x_relu_pool + 90,
            pool_y - 12,
            x_relu_pool + 110,
            pool_y + 12,
            fill="#A0A0A0",
        )
        self.elements[("relu_pool", "pool_arrow")] = arrow_id

        # Flatten: small 2D box -> arrow -> line of dots
        x_flatten = self.x_positions["flatten"]
        self.canvas.create_text(
            x_flatten,
            self.top_y,
            text="Flatten",
            font=("Helvetica", 14, "bold"),
            justify="center",
            fill="#333",
        )
        # 2D box
        flat_box = self.canvas.create_rectangle(
            x_flatten - 40,
            self.y_center - 20,
            x_flatten - 10,
            self.y_center + 20,
            fill="#808080",
        )
        # arrow
        flat_arrow = self.canvas.create_line(
            x_flatten - 5,
            self.y_center,
            x_flatten + 20,
            self.y_center,
            arrow="last",
            width=2,
        )
        # dots
        dot_x_start = x_flatten + 30
        for d in range(5):
            self.canvas.create_oval(
                dot_x_start + d * 10 - 3,
                self.y_center - 3,
                dot_x_start + d * 10 + 3,
                self.y_center + 3,
                fill="black",
            )

        # FC Layer: 10 neurons
        x_fc = self.x_positions["fc"]
        fc_neurons = 10
        dot_radius = 15
        spacing = 10
        total_height = fc_neurons * (2 * dot_radius + spacing) - spacing
        start_y = self.y_center - total_height / 2 + dot_radius

        self.canvas.create_text(
            x_fc,
            self.top_y,
            text="FC\n(10 neurons)",
            font=("Helvetica", 14, "bold"),
            justify="center",
            fill="#333",
        )

        for i in range(fc_neurons):
            cy = start_y + i * (2 * dot_radius + spacing)
            n_id = self.canvas.create_oval(
                x_fc - dot_radius,
                cy - dot_radius,
                x_fc + dot_radius,
                cy + dot_radius,
                fill="white",
                outline="black",
                width=1,
            )
            self.elements[("fc", f"neuron_{i}")] = n_id

        # Softmax predicted class and actual label
        x_softmax = self.x_positions["softmax"]
        self.canvas.create_text(
            x_softmax,
            self.top_y,
            text="Output\n(Softmax)",
            font=("Helvetica", 14, "bold"),
            justify="center",
            fill="#333",
        )

        softmax_center_x = x_softmax + 200  # shift bars & text to the right a bit
        softmax_text_y = self.y_center

        class_text_id = self.canvas.create_text(
            softmax_center_x,
            softmax_text_y - 60,
            text="Predicted: ...",
            font=("Helvetica", 12),
            fill="black",
            anchor="w",
        )
        self.elements[("softmax", "class_text")] = class_text_id

        label_text_id = self.canvas.create_text(
            softmax_center_x,
            softmax_text_y - 40,
            text="Label: ...",
            font=("Helvetica", 12),
            fill="black",
            anchor="w",
        )
        self.elements[("softmax", "label_text")] = label_text_id

        # Top-k classes just below predicted/label
        for i in range(3):
            t_id = self.canvas.create_text(
                softmax_center_x,
                softmax_text_y - 20 + i * 20,
                text=f"Top {i+1}: ...",
                font=("Helvetica", 12),
                fill="black",
                anchor="w",
            )
            self.elements[("softmax", f"topk_class_{i}")] = t_id

        # Probability bar chart
        # We'll stack bars vertically, class 0 at top, class 9 at bottom
        # each bar on the right of these texts
        bar_x_start = softmax_center_x - 200
        bar_y_start = softmax_text_y + 100
        bar_width = 100
        bar_height_max = 50
        bar_spacing = 5

        # store these for dynamic to know top coords
        self.bar_width = bar_width
        self.bar_height_max = bar_height_max
        self.bar_x_start = bar_x_start
        self.bar_y_start = bar_y_start - 350  # move up a bit
        self.bar_spacing = bar_spacing  # reduce spacing

        for c in range(10):
            cy = bar_y_start + c * (bar_height_max + bar_spacing)
            # Initially zero-length
            bar_id = self.canvas.create_rectangle(
                bar_x_start - 150,
                cy,
                bar_x_start + 1,
                cy,  # tiny line
                fill="#00A0A0",
                outline="black",
            )
            self.elements[("softmax", f"class_bar_{c}")] = bar_id
            label_id = self.canvas.create_text(
                bar_x_start + bar_width + 30,
                cy - 300,
                text=f"{c}: 0.00%",
                font=("Helvetica", 12),
                fill="black",
                anchor="w",
            )
            self.elements[("softmax", f"class_bar_label_{c}")] = label_id
