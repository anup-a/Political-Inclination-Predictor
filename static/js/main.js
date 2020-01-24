$(document).ready(function () {
    // Init
    $('form').bind("keypress", function(e) {
        if (e.keyCode == 13) {               
          e.preventDefault();
          return false;
        }
      });
      
    // $('.')
    $('.loader').hide();
    $('#result').hide();

    $('.btn-custom').click(function () {

        var form_data = new FormData($('.upload-file')[0]);

        text = $(".active div.jumbotron form.upload-file input#input-msg").val();

        form_data.append('active',($('.active').first()).text());
        form_data.append('text', $(".active div.jumbotron form.upload-file input#input-msg").val());

        $('.loader').show();

        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                let str_len = data['party'].length
                let prob_arr = data['party'].slice(2,str_len-2).split(" ")
                let fin_arr = []
                let indices = []
                for(let i=0;i<prob_arr.length;++i){
                    if (prob_arr[i].length > 0){
                        fin_arr.push(parseFloat(prob_arr[i]))
                        if(0.5 < fin_arr[i]){
                            indices.push(i)
                        }
                    }
                }
                // let max_index = fin_arr.indexOf(Math.max(fin_arr));
                let party_arr = ['INC-Against','Others','INC-Favour','BJP-Favour','BJP-Against']
                let html_string = ""
                for(let i=0;i<indices.length;++i){
                    html_string += '</span>'+"<span class='party'>"+party_arr[indices[i]]+'</span>'
                }
                console.log(fin_arr)
                $('.loader').fadeOut(600);
                $('#result').fadeIn(1000);
                var isPolitical = (data['isPolitical']).toString();
                $("input:text").val('');
                if ($('ul li').length >= 7){
                    $('ul li:last-child').remove();
                }
                if(isPolitical == "Political"){
                    $('ul').prepend('<li class="list-group-item">'+ text  +"<span class="+isPolitical+">"+isPolitical+html_string+'</li>');
                } else {
                    $('ul').prepend('<li class="list-group-item">'+ text  +"<span class="+isPolitical+">"+isPolitical+'</span></li>');
                }
                console.log('Success!');
            },
        });
    });

});
