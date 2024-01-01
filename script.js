(async () => {
    const model = await tf.loadLayersModel('My Model/My Model.json');

    // const input_layer = tf.input({shape: [2]});
    // const hidden_layer1 = tf.layers.dense({units: 3, activation: 'relu'}).apply(input_layer);
    // const hidden_layer2 = tf.layers.dense({units: 3, activation: 'relu'}).apply(hidden_layer1);
    // const output_layer = tf.layers.dense({units: 1, activation: 'sigmoid'}).apply(hidden_layer2);
  
    // const model = tf.model({inputs: input_layer, outputs: output_layer});

    // const compileParam = { optimizer: tf.train.adam(), loss: 'binaryCrossentropy' };

    // model.compile(compileParam);

    // const Input_Arary = [[0, 0], [0, 1], [1, 0], [1, 1]];
    // const Output_Array = [[0], [1], [1], [0]];

    const Test_Input_Array = [[0, 0], [0, 1], [1, 0], [1, 1]];
  
    // const xs = tf.tensor2d(Input_Arary, [Input_Arary.length, 2]);
    // const ys = tf.tensor2d(Output_Array, [Output_Array.length, 1]);

    // let fitParam = {
    //     epochs: 10000,
    //     callbacks: {
    //         onEpochEnd: (epoch, logs) => {
    //             document.getElementById('output').innerHTML = `epoch = ${epoch}, Loss = ${logs.loss}`;
    //         }
    //     }
    // };

    // await model.fit(xs, ys, fitParam);

    const predictions = model.predict(tf.tensor2d(Test_Input_Array, [Test_Input_Array.length, 2])).dataSync();
    console.log(predictions);

    // await model.save('downloads://My Model');
})();