from PIL import Image, ImageTk
import numpy as np


class DiagramDynamic:
    def __init__(self, canvas, layout):
        self.canvas = canvas
        self.layout = layout

    def update_input_image(self, new_array):
        if new_array.ndim == 3 and new_array.shape[0] == 1:
            new_array = new_array[0]

        pil_img = Image.fromarray(new_array, mode="L")
        pil_img = pil_img.resize((280, 280), Image.NEAREST)
        new_photo = ImageTk.PhotoImage(pil_img)

        self.layout.input_photo = new_photo

        if ("input", "image") in self.layout.elements:
            image_id = self.layout.elements[("input", "image")]
            self.canvas.itemconfigure(image_id, image=new_photo)

    def update_values(self, intermediates, label):
        # Update input mean
        if ("input", "text") in self.layout.elements:
            text_id = self.layout.elements[("input", "text")]
            input_data = intermediates["input"]
            input_mean = input_data.mean()
            self.canvas.itemconfigure(text_id, text=f"Val={input_mean:.2f}")

        # Conv filters
        if "conv" in intermediates:
            conv_out = intermediates["conv"]
            _, C, H_out, W_out = conv_out.shape
            scale_factor = min(70 / W_out, 70 / H_out)
            for i in range(C):
                filter_map = conv_out[0, i, :, :]
                f_min, f_max = filter_map.min(), filter_map.max()
                if f_max > f_min:
                    norm_map = (filter_map - f_min) / (f_max - f_min) * 255.0
                else:
                    norm_map = np.full_like(filter_map, 127.0)
                norm_map = norm_map.astype(np.uint8)

                new_w = int(W_out * scale_factor)
                new_h = int(H_out * scale_factor)
                filter_img = Image.fromarray(norm_map, mode="L")
                filter_img = filter_img.resize((new_w, new_h), Image.NEAREST)
                filter_photo = ImageTk.PhotoImage(filter_img)

                key = ("conv", f"filter_{i}")
                if key in self.layout.elements:
                    filter_id = self.layout.elements[key]
                    self.canvas.itemconfigure(filter_id, image=filter_photo)
                if not hasattr(self, "conv_filter_photos"):
                    self.conv_filter_photos = {}
                self.conv_filter_photos[i] = filter_photo

        # FC layer
        if "fc" in intermediates:
            fc_out = intermediates["fc"][0]
            f_min, f_max = fc_out.min(), fc_out.max()
            diff = f_max - f_min if f_max > f_min else 1
            # Reset outlines
            for i in range(10):
                key = ("fc", f"neuron_{i}")
                if key in self.layout.elements:
                    n_id = self.layout.elements[key]
                    self.canvas.itemconfigure(n_id, outline="black", width=1)
            # Update color intensity
            for i in range(10):
                val_norm = (fc_out[i] - f_min) / diff
                intensity = int(val_norm * 255)
                color = f"#{intensity:02x}{intensity:02x}{intensity:02x}"
                key = ("fc", f"neuron_{i}")
                if key in self.layout.elements:
                    n_id = self.layout.elements[key]
                    self.canvas.itemconfigure(n_id, fill=color)

        # Softmax and top-k
        if "softmax" in intermediates:
            softmax_out = intermediates["softmax"][0]
            pred_class = np.argmax(softmax_out)

            if ("softmax", "class_text") in self.layout.elements:
                class_text_id = self.layout.elements[("softmax", "class_text")]
                self.canvas.itemconfigure(
                    class_text_id, text=f"Predicted: {pred_class}"
                )

            if ("softmax", "label_text") in self.layout.elements:
                label_text_id = self.layout.elements[("softmax", "label_text")]
                self.canvas.itemconfigure(label_text_id, text=f"Label: {label}")

            k = 3
            topk_indices = np.argsort(softmax_out)[::-1][:k]
            for i, idx in enumerate(topk_indices):
                prob = softmax_out[idx] * 100
                font_style = ("Helvetica", 12, "bold") if i == 0 else ("Helvetica", 12)
                text = f"Top {i+1}: Class {idx}, {prob:.2f}%"
                tk_key = ("softmax", f"topk_class_{i}")
                if tk_key in self.layout.elements:
                    tk_id = self.layout.elements[tk_key]
                    self.canvas.itemconfigure(tk_id, text=text, font=font_style)

            # Probability bar chart
            bar_height_max = self.layout.bar_height_max
            bar_spacing = self.layout.bar_spacing
            bar_x_start = self.layout.bar_x_start
            bar_width = self.layout.bar_width
            bar_y_start = self.layout.bar_y_start

            for c in range(10):
                cy = bar_y_start + c * (bar_height_max + bar_spacing)
                prob = softmax_out[c]
                bar_height = int(prob * bar_height_max)
                # we decided top: cy+(bar_height_max - bar_height), bottom: cy+bar_height_max
                bar_top = cy + (bar_height_max - bar_height)
                bar_bottom = cy + bar_height_max
                bar_key = ("softmax", f"class_bar_{c}")
                if bar_key in self.layout.elements:
                    bar_id = self.layout.elements[bar_key]
                    self.canvas.coords(
                        bar_id,
                        bar_x_start,
                        bar_top,
                        bar_x_start + bar_width,
                        bar_bottom,
                    )

                label_key = ("softmax", f"class_bar_label_{c}")
                if label_key in self.layout.elements:
                    label_id = self.layout.elements[label_key]
                    self.canvas.itemconfigure(label_id, text=f"{c}: {prob*100:.2f}%")

            # Compare predicted class with actual label
            correct = pred_class == label
            pred_key = ("fc", f"neuron_{pred_class}")
            if pred_key in self.layout.elements:
                pred_id = self.layout.elements[pred_key]
                outline_color = "green" if correct else "red"
                self.canvas.itemconfigure(pred_id, outline=outline_color, width=2)

            if not correct:
                correct_key = ("fc", f"neuron_{label}")
                if correct_key in self.layout.elements:
                    correct_id = self.layout.elements[correct_key]
                    self.canvas.itemconfigure(correct_id, outline="blue", width=2)
