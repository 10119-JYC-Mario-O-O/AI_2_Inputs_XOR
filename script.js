(async () => {
  // const model = await tf.loadLayersModel('My_Model/m y-model.json');

  // Create a simple model.
  const input_layer = tf.input({shape: [2]});
  const hidden_layer1 = tf.layers.dense({units: 3, activation: 'relu'}).apply(input_layer);
  const hidden_layer2 = tf.layers.dense({units: 3, activation: 'relu'}).apply(hidden_layer1);
  const output_layer = tf.layers.dense({units: 1, activation: 'softmax'}).apply(hidden_layer2);
  
  const model = tf.model({inputs: input_layer, outputs: output_layer});

  // Prepare the model for training: Specify the loss and the optimizer.
  const compileParam = { optimizer: tf.train.adam(), loss: tf.losses.meanSquaredError };

  model.compile(compileParam);

  tf.show.modelSummary({name:'summary', tab:'model'},model);

  // Generate some synthetic data for training.
  const Input_Arary = [[0, 0], [0, 1], [1, 0], [1, 1]];
  const Output_Array = [[0], [1], [1], [0]];

  const Test_Input_Array = [[0, 0], [0, 1], [1, 0], [1, 1]];
  
  const xs = tf.tensor2d(Input_Arary, [Input_Arary.length, 2]);
  const ys = tf.tensor2d(Output_Array, [Output_Array.length, 1]);

  // Train the model using the data.
  let fitParam = {
    epochs: 10000,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        document.getElementById('micro-out-div').innerHTML = `epoch = ${epoch}, RMSE = ${Math.sqrt(logs.loss)}`;
        _history.push(logs);
        tfvis.show.history({name:'loss', tab:'history'}, _history, ['loss'])
      }
    }
  };

  await model.fit(xs, ys, fitParam);

  // Use the model to do inference on a data point the model hasn't seen.
  // Should print approximately.
  // document.getElementById('micro-out-div').innerText = model.predict(tf.tensor2d([[0, 0]], [1, 2])).dataSync();

  console.log(model.predict(tf.tensor2d(Test_Input_Array, [Test_Input_Array.length, 2])).dataSync());

  model.save('downloads://my-model');
})();