from PIL import Image
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.visualization.petri_net import visualizer as pn_visualizer


def convert_xes_to_pnml_and_image(xes_path: str, pnml_path: str, image_path: str):
    # Step 1: Import the XES event log
    log = xes_importer.apply(xes_path)

    # Step 2: Discover the Petri net using the Alpha Miner
    net, initial_marking, final_marking = alpha_miner.apply(log)

    # Step 3: Export the discovered Petri net to PNML
    pnml_exporter.apply(net, initial_marking, pnml_path, final_marking=final_marking)
    print(f"Petri net saved to: {pnml_path}")

    # Step 4: Visualize and export as image
    gviz = pn_visualizer.apply(net, initial_marking, final_marking)
    pn_visualizer.save(gviz, image_path)
    convert_png_to_pdf("reference_model.png", "reference_model.pdf")
    print(f"Petri net image saved to: {image_path}")

def convert_png_to_pdf(png_path, pdf_path):
    image = Image.open(png_path)
    if image.mode == "RGBA":
        image = image.convert("RGB")
    image.save(pdf_path, "PDF", resolution=100.0)
    print(f"Image converted to PDF: {pdf_path}")


# Example usage:
if __name__ == "__main__":
    convert_xes_to_pnml_and_image("Event log for reference model.xes", "reference_model.pnml", "reference_model.png")
