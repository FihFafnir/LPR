import glob
import matplotlib.pyplot as plot


PATH = "./treated_images/"

cars = sorted(glob.glob(PATH + "car_*.jpg"))
treated_cars = sorted(glob.glob(PATH + "treated_car_*.jpg"))
plates = sorted(glob.glob(PATH + "plate_*.jpg"))
treated_plates = sorted(glob.glob(PATH + "treated_plate_*.jpg"))


def multiplot(image_1, image_2, image_3, image_4):
    fig = plot.figure(figsize=(10, 7))

    # setting values to rows and column variables
    rows = 2
    columns = 2

    image_1, image_2, image_3, image_4 = map(
        plot.imread, (image_1, image_2, image_3, image_4)
    )

    # Adds a subplot at the 1st position
    fig.add_subplot(rows, columns, 1)

    # showing image
    plot.imshow(image_1, cmap="gray")
    plot.axis("off")
    plot.title("1 - Imagem de entrada sem tratamento.")

    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 2)

    # showing image
    plot.imshow(image_2, cmap="gray")
    plot.axis("off")
    plot.title("2 - Pós-tratamento para extração da placa.")

    # Adds a subplot at the 3rd position
    fig.add_subplot(rows, columns, 3)

    # showing image
    plot.imshow(image_3, cmap="gray")
    plot.axis("off")
    plot.title("3 - Imagem da placa extraída.")

    # Adds a subplot at the 4th position
    fig.add_subplot(rows, columns, 4)

    # showing image
    plot.imshow(image_4, cmap="gray")
    plot.axis("off")
    plot.title("4 - Placa tratada para leitura.")

    plot.savefig(
        f"./treated_images/plot_{len(glob.glob('./treated_images/plot_*.jpg')) + 1}.jpg"
    )
    # plot.show()


for i in range(len(cars)):
    multiplot(cars[i], treated_cars[i], plates[i], treated_plates[i])
