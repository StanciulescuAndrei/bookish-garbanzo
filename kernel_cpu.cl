void inBounds(int x, int y, int width, int height){
	return (x >=0 ) || (y >= 0) || (x < width) || (y < height);
}


__kernel void ThermalPropagation(__global float * input_image, __global float * output_image,__constant float * data){
    int width = data[0];
    int height = data[1];
    float sourceTemp = data[2];
    float extTemp = data[3];
	int2 sourceCoords = (int2)(data[4], data[5]);
	float ratio = 0.0F;
	float extTransfer = 1.0F;

    int2 coords = (int2)(get_global_id(0), get_global_id(1));

    float color = 0.0F;

    if ((coords.x >= width) || (coords.y >= height))
	{
		return;
	}

	//Edge case
	if(coords.x == 0 || coords.y == 0 || coords.x == width - 1 || coords.y == height - 1){
		if(coords.x == 0)
			output_image[coords.y * width + coords.x + 1] = extTemp;
		else if(coords.y == 0)
			output_image[(coords.y + 1) * width + coords.x] = extTemp;
		else if(coords.x == width - 1)
			output_image[coords.y * width + coords.x - 1] = extTemp;
		else if(coords.y == height - 1)
			output_image[(coords.y - 1) * width + coords.x] = extTemp;
		//output_image[coords.y * width + coords.x] = extTemp;
		return;
	}

	if(coords.x == sourceCoords.x && coords.y == sourceCoords.y){
		output_image[coords.y * width + coords.x] = sourceTemp;
		return;
	}

	//Point is inside the plate
	for(int dx = -1; dx < 2; dx++){
		for(int dy = -1; dy < 2; dy++){
			if(coords.x + dx == 0 || coords.y + dy == 0 || coords.x + dx == width - 1 || coords.y + dy == height - 1){
				color += (input_image[coords.x + dx + (coords.y + dy) * width] * extTransfer);
				ratio += extTransfer;
			}
			else{
				color += (input_image[coords.x + dx + (coords.y + dy) * width]);
				ratio += 1.0F;
			}
		}
	}
	color /= ratio;

	output_image[coords.y * width + coords.x] = color;
    

}