
__kernel void ThermalPropagation(__read_only image2d_t input_image, __write_only image2d_t output_image,__constant float * data, sampler_t sampler){
    int width = data[0];
    int height = data[1];
    float sourceTemp = data[2];
    float extTemp = data[3];
	int2 sourceCoords = (int2)(data[4], data[5]);
	float ratio = 0.0F;

    int2 coords = (int2)(get_global_id(0), get_global_id(1));

    float4 color = (float4)(0.0F, 0.0F, 0.0F, 0.0F);
	
    if ((coords.x >= width) || (coords.y >= height))
	{
		return;
	}

	if(coords.x == sourceCoords.x && coords.y == sourceCoords.y){
		write_imagef(output_image, coords, (float4)(sourceTemp, 0.0F, 0.0F, 0.0F));
		return;
	}

	//Point is inside the plate
	for(int dx = -1; dx < 2; dx++){
		for(int dy = -1; dy < 2; dy++){
			color += read_imagef(input_image, sampler, (int2)(coords.x + dx, coords.y + dy));
			ratio += 1.0F;
		}
	}
	color /= ratio;

	write_imagef(output_image, coords, color);
    

}